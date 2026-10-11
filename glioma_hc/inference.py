from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from glioma_hc.config import SETTINGS
from glioma_hc.dataset_builder.preprocessing import COMMON_IMAGE_SIZE, preprocess_mri_rgb
from glioma_hc.evaluation.reporting import (
    is_current_ensemble_test_report,
    is_current_single_model_test_report,
)
from glioma_hc.explainability.gradcam import GradCAMUnavailable, make_gradcam_overlay
from glioma_hc.explainability.guided_backprop import (
    GuidedBackpropUnavailable,
    encode_attribution,
)
from glioma_hc.explainability.hybrid_cam import HybridCAMUnavailable, make_hybrid_cam
from glioma_hc.training.data_loader import preprocess_model_input
from glioma_hc.training.ensemble import combine_model_probabilities
from glioma_hc.training.ensemble import deployment_from_training_summary
from glioma_hc.training.model_zoo import (
    ChannelPool2D,
    MODEL_SPECS,
    REQUIRED_ENSEMBLE_MODELS,
)


GRADCAM_LAYERS = {
    "resnet18": "target_conv_activation",
    "resnet50": "conv5_block3_out",
    "resnet101": "conv5_block3_out",
    "googlenet": "target_conv_activation",
    "efficientnet_b0": "top_activation",
    "mobilenet_v2": "out_relu",
    "inception_v3": "mixed10",
    "densenet121": "conv5_block16_concat",
    "kaggle_cnn": "target_conv_activation",
}


class InferenceUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_KEY: tuple[Any, ...] | None = None
_MODEL_CACHE: tuple[list[tuple[str, Any]], dict[str, Any]] | None = None


def clear_model_cache() -> None:
    global _MODEL_CACHE_KEY, _MODEL_CACHE
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE_KEY = None
        _MODEL_CACHE = None


def _load_keras_model(path: Path) -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise InferenceUnavailable("TensorFlow is unavailable in the server environment.") from exc
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "ChannelPool2D": ChannelPool2D,
            "GliomaScanHC>ChannelPool2D": ChannelPool2D,
        },
    )


def _load_eligible_models() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    global _MODEL_CACHE_KEY, _MODEL_CACHE
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    if not summary_path.is_file():
        raise InferenceUnavailable("No completed eight-model training run was recorded.")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceUnavailable("The training summary is unavailable or invalid.") from exc

    if summary.get("status") != "ok" or summary.get("quick") is not False:
        raise InferenceUnavailable("Predictions require a completed non-smoke training run.")
    if summary.get("label_order") != SETTINGS.label_order:
        raise InferenceUnavailable(
            "The trained model class order does not match the configured labels; retrain the ensemble."
        )
    if summary.get("ensemble", {}).get("selection_split") != "val":
        raise InferenceUnavailable("The ensemble strategy must be selected using validation data.")

    records = {
        record.get("name"): record
        for record in summary.get("models", [])
        if isinstance(record, dict)
    }
    missing = [name for name in REQUIRED_ENSEMBLE_MODELS if name not in records]
    if missing:
        raise InferenceUnavailable(
            "The required eight-CNN ensemble is incomplete. Train missing models: "
            + ", ".join(missing)
        )
    ensemble = summary["ensemble"]
    strategy = ensemble.get("strategy")
    if strategy not in {"hard_voting", "soft_voting", "weighted_soft_voting"}:
        raise InferenceUnavailable("No validation-selected ensemble strategy is available.")
    if ensemble.get("model_order") != list(REQUIRED_ENSEMBLE_MODELS):
        raise InferenceUnavailable("The ensemble model order does not match the required eight CNNs.")

    deployment = deployment_from_training_summary(summary)
    if deployment is None:
        raise InferenceUnavailable("No validation-selected deployment candidate is available.")
    if deployment.get("selection_split") != "val":
        raise InferenceUnavailable("The deployment candidate must be selected on validation data.")
    if deployment.get("kind") == "single_model":
        model_name = deployment.get("model")
        if model_name not in MODEL_SPECS:
            raise InferenceUnavailable("The validation-selected model is unsupported.")
        if deployment.get("strategy") != "single_model":
            raise InferenceUnavailable("The validation-selected model strategy is invalid.")
        preprocessing_profile = deployment.get("preprocessing_profile", "shared_mri")
        if preprocessing_profile not in {"shared_mri", "kaggle_notebook_v1"}:
            raise InferenceUnavailable("The selected model preprocessing profile is unsupported.")
        if preprocessing_profile == "kaggle_notebook_v1" and model_name != "kaggle_cnn":
            raise InferenceUnavailable("The legacy notebook preprocessing profile only supports Kaggle CNN.")
        names_to_load = [model_name]
    elif deployment.get("kind") == "ensemble":
        if deployment.get("strategy") != strategy:
            raise InferenceUnavailable("The validation-selected ensemble strategy is inconsistent.")
        preprocessing_profile = "shared_mri"
        names_to_load = list(REQUIRED_ENSEMBLE_MODELS)
    else:
        raise InferenceUnavailable("The validation-selected deployment candidate is invalid.")

    checkpoint_paths = {}
    if deployment.get("kind") == "single_model":
        name = names_to_load[0]
        if name in REQUIRED_ENSEMBLE_MODELS:
            record = records[name]
            model_path = Path(record.get("model_path", "")).resolve()
            allowed_root = SETTINGS.models_dir.resolve()
            if record.get("status") != "trained":
                raise InferenceUnavailable(f"The {name} model is not marked as trained.")
        else:
            model_path = Path(deployment.get("model_path", "")).resolve()
            allowed_root = (SETTINGS.project_root / "models").resolve()
        if (
            model_path.suffix != ".keras"
            or not model_path.is_relative_to(allowed_root)
            or not model_path.is_file()
        ):
            raise InferenceUnavailable(f"The selected {name} checkpoint is missing or outside its allowed model directory.")
        checkpoint_paths[name] = model_path
    else:
        models_root = SETTINGS.models_dir.resolve()
        for name in REQUIRED_ENSEMBLE_MODELS:
            record = records[name]
            model_path = Path(record.get("model_path", "")).resolve()
            if record.get("status") != "trained" or model_path.suffix != ".keras":
                raise InferenceUnavailable(f"The {name} model is not marked as trained.")
            if not model_path.is_relative_to(models_root) or not model_path.is_file():
                raise InferenceUnavailable(f"The saved {name} checkpoint is missing or outside saved_models/.")
            checkpoint_paths[name] = model_path

    summary_stat = summary_path.stat()
    checkpoint_signature = tuple(
        (
            name,
            str(path),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for name, path in checkpoint_paths.items()
    )
    cache_key = (
        str(summary_path.resolve()),
        summary_stat.st_mtime_ns,
        summary_stat.st_size,
        checkpoint_signature,
        tuple(names_to_load),
        preprocessing_profile,
    )

    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE_KEY == cache_key and _MODEL_CACHE is not None:
            loaded_models, cached_summary = _MODEL_CACHE
            return loaded_models, cached_summary

        loaded_models = []
        for name in names_to_load:
            try:
                model = _load_keras_model(checkpoint_paths[name])
            except Exception as exc:
                raise InferenceUnavailable(f"Could not load trained model {name}.") from exc
            if not isinstance(model.input_shape, tuple) or len(model.input_shape) != 4:
                raise InferenceUnavailable(f"The {name} model has an unsupported input shape.")
            if not isinstance(model.output_shape, tuple) or model.output_shape[-1] != len(SETTINGS.label_order):
                raise InferenceUnavailable(f"The {name} model has an unsupported output shape.")
            loaded_models.append((name, model))

        cached_summary = {**summary, "deployment": deployment}
        _MODEL_CACHE_KEY = cache_key
        _MODEL_CACHE = (loaded_models, cached_summary)
        return loaded_models, cached_summary


def _encode_rgb_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{b64encode(buffer.getvalue()).decode('ascii')}"


def _preprocess_uploaded_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    common_image = image.convert("RGB").resize(
        (COMMON_IMAGE_SIZE[1], COMMON_IMAGE_SIZE[0]),
        Image.Resampling.BICUBIC,
    )
    original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    processed_common = preprocess_mri_rgb(np.asarray(common_image, dtype=np.uint8))
    return original_rgb, processed_common


def predict_images(
    image_files: list[tuple[str, bytes]],
    aggregation: str = "max",
    explain: str = "hybrid_cam",
    sequence: str = "UNSPECIFIED",
) -> dict[str, Any]:
    if not image_files:
        raise InvalidImage("Upload at least one raster image.")
    if aggregation not in {"max", "mean"}:
        raise InvalidImage("Aggregation must be 'max' or 'mean'.")
    if explain not in {"gradcam", "hybrid_cam", "none"}:
        raise InvalidImage("Explain must be 'gradcam', 'hybrid_cam', or 'none'.")

    images = []
    for filename, content in image_files:
        try:
            with Image.open(BytesIO(content)) as image:
                original_rgb, processed_common = _preprocess_uploaded_image(image)
                images.append({
                    "filename": filename,
                    "image": image.convert("RGB").copy(),
                    "original_rgb": original_rgb,
                    "processed_common": processed_common,
                    "original_size": image.size,
                })
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImage(f"{filename} is not a supported raster image.") from exc

    models, summary = _load_eligible_models()
    ensemble_config = summary["ensemble"]
    deployment = summary["deployment"]
    strategy = deployment["strategy"]
    predictor_kind = deployment["kind"]
    preprocessing_profile = deployment.get("preprocessing_profile", "shared_mri")
    weights_by_name = ensemble_config.get("model_weights", {})
    if predictor_kind == "single_model":
        strategy = "single_model"
        weights = np.ones(1, dtype=np.float64)
        weights_by_name = {models[0][0]: 1.0}
    else:
        weights = np.asarray(
            [weights_by_name.get(name, 0.0) for name, _ in models],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise InferenceUnavailable("Validation-derived model weights are invalid.")

    per_model_scores: list[np.ndarray] = []
    model_inputs: dict[str, np.ndarray] = {}
    elapsed_times = {}
    for name, model in models:
        _, height, width, channels = model.input_shape
        if not all(isinstance(size, int) and size > 0 for size in (height, width)) or channels != 3:
            raise InferenceUnavailable(f"Model {name} has an unsupported input shape.")
        if preprocessing_profile == "kaggle_notebook_v1":
            import tensorflow as tf

            resized_images = np.stack([
                tf.image.resize(
                    item["original_rgb"],
                    (height, width),
                    method="bilinear",
                ).numpy()
                for item in images
            ]).astype(np.float32)
        else:
            resized_images = np.stack([
                cv2.resize(
                    item["processed_common"],
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                for item in images
            ]).astype(np.float32)
        model_batch = preprocess_model_input(resized_images, name)
        model_inputs[name] = model_batch

        started = time.perf_counter()
        scores = np.asarray(model.predict(model_batch, verbose=0), dtype=np.float64)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if scores.shape != (len(images), len(SETTINGS.label_order)):
            raise InferenceUnavailable(f"Model {name} returned an unsupported prediction shape.")
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise InferenceUnavailable(f"Model {name} returned invalid probabilities.")
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
            raise InferenceUnavailable(f"Model {name} outputs are not normalized probabilities.")

        elapsed_times[name] = round(elapsed_ms, 3)
        per_model_scores.append(scores)

    model_probability_tensor = np.stack(per_model_scores)
    if predictor_kind == "single_model":
        ensemble_scores = model_probability_tensor[0]
    else:
        ensemble_scores = combine_model_probabilities(
            model_probability_tensor,
            strategy,
            weights if strategy == "weighted_soft_voting" else None,
        )
    if aggregation == "max":
        selected_slice_index = int(np.argmax(ensemble_scores.max(axis=1)))
        class_probabilities = ensemble_scores[selected_slice_index]
    else:
        class_probabilities = ensemble_scores.mean(axis=0)
        class_probabilities /= class_probabilities.sum()
        selected_slice_index = int(np.argmax(ensemble_scores[:, np.argmax(class_probabilities)]))

    predicted_class_index = int(np.argmax(class_probabilities))
    label = SETTINGS.label_order[predicted_class_index]
    confidence = float(class_probabilities[predicted_class_index])
    per_model_predictions = []
    for model_index, (name, _) in enumerate(models):
        model_probabilities = per_model_scores[model_index][selected_slice_index]
        model_label_index = int(np.argmax(model_probabilities))
        per_model_predictions.append({
            "name": name,
            "label": SETTINGS.label_order[model_label_index],
            "confidence": float(model_probabilities[model_label_index]),
            "probs": model_probabilities.tolist(),
            "class_probabilities": {
                class_name: float(model_probabilities[class_index])
                for class_index, class_name in enumerate(SETTINGS.label_order)
            },
            "inference_ms": elapsed_times[name],
            "agrees_with_ensemble": model_label_index == predicted_class_index,
        })
    agreeing_models = sum(item["agrees_with_ensemble"] for item in per_model_predictions)
    vote_counts = {
        class_name: sum(item["label"] == class_name for item in per_model_predictions)
        for class_name in SETTINGS.label_order
    }

    explanation_model_name = (
        models[0][0]
        if predictor_kind == "single_model"
        else max(
            REQUIRED_ENSEMBLE_MODELS,
            key=lambda name: weights_by_name.get(name, 0.0),
        )
    )
    explanation_model = dict(models)[explanation_model_name]
    explanation_index = selected_slice_index
    explanation_image = images[explanation_index]["original_rgb"]
    explanation_input = model_inputs[explanation_model_name][
        explanation_index:explanation_index + 1
    ]
    gradcam_overlay = None
    guided_backprop_image = None
    hybrid_cam_overlay = None
    explanation_status = "disabled" if explain == "none" else "unavailable"
    explanation_reason = "Explainability was not requested." if explain == "none" else ""
    fusion_method = None
    attribution_overlap = None
    if explain in {"gradcam", "hybrid_cam"}:
        layer_name = GRADCAM_LAYERS[explanation_model_name]
        try:
            gradcam_overlay = make_gradcam_overlay(
                explanation_model,
                explanation_input,
                explanation_image,
                layer_name=layer_name,
                class_index=predicted_class_index,
            )
            if explain == "hybrid_cam":
                hybrid_result = make_hybrid_cam(
                    explanation_model,
                    explanation_input,
                    explanation_image,
                    predicted_class_index,
                    layer_name=layer_name,
                )
                guided_backprop_image = encode_attribution(hybrid_result["guided_backprop"])
                hybrid_cam_overlay = hybrid_result["overlay"]
                attribution_overlap = hybrid_result["attribution_overlap"]
                fusion_method = hybrid_result["fusion"]
            explanation_status = "generated"
        except (GradCAMUnavailable, HybridCAMUnavailable, GuidedBackpropUnavailable) as exc:
            explanation_reason = str(exc)

    if predictor_kind == "single_model":
        metrics_path = SETTINGS.reports_dir / "kaggle_cnn_paper_test.json"
    else:
        metrics_path = SETTINGS.reports_dir / "weighted_soft_ensemble_test.json"
        if not metrics_path.is_file():
            metrics_path = SETTINGS.reports_dir / "final_test_results.json"
        if not metrics_path.is_file():
            metrics_path = SETTINGS.reports_dir / "test_results.json"
    evaluation_metrics = {}
    if metrics_path.is_file():
        try:
            evaluation_report = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceUnavailable("The saved held-out evaluation report is invalid.") from exc
        if (
            is_current_single_model_test_report(evaluation_report, deployment)
            if predictor_kind == "single_model"
            else is_current_ensemble_test_report(evaluation_report, summary)
        ):
            evaluation_metrics = evaluation_report.get("metrics", {})

    selected_image = images[selected_slice_index]
    return {
        "id": str(uuid4()),
        "label": label,
        "class_probabilities": {
            class_name: float(class_probabilities[index])
            for index, class_name in enumerate(SETTINGS.label_order)
        },
        "glioma_probability": float(
            class_probabilities[SETTINGS.label_order.index("glioma")]
        ),
        "confidence": confidence,
        "aggregation": aggregation,
        "ensemble": {
            "strategy": strategy,
            "selection_split": "validation",
            "confidence": confidence,
            "models_agreeing": agreeing_models,
            "total_models": len(models),
            "vote_counts": vote_counts,
        },
        "predictor": {
            "kind": predictor_kind,
            "model": deployment.get("model"),
            "selection_split": "validation",
        },
        "models": per_model_predictions,
        "slices": [
            {
                "index": index,
                "sequence": sequence.upper(),
                "used_for_classification": True,
                "label": SETTINGS.label_order[int(np.argmax(score))],
                "class_probabilities": {
                    class_name: float(score[class_index])
                    for class_index, class_name in enumerate(SETTINGS.label_order)
                },
            }
            for index, score in enumerate(ensemble_scores)
        ],
        "input": {
            "filename": selected_image["filename"],
            "modality": "Brain MRI (user-provided raster; not independently verified)",
            "original_width": int(selected_image["original_size"][0]),
            "original_height": int(selected_image["original_size"][1]),
            "common_preprocessing_size": (
                None
                if preprocessing_profile == "kaggle_notebook_v1"
                else list(COMMON_IMAGE_SIZE)
            ),
            "preprocessed_model_inputs": {
                name: list(model.input_shape[1:3])
                for name, model in models
            },
        },
        "preprocessing": {
            "profile": preprocessing_profile,
            "steps": (
                [
                    "Convert the uploaded raster to RGB.",
                    "Resize the RGB image directly to the Kaggle CNN's 150 x 150 input.",
                    "Scale pixel values to [0, 1], matching the saved notebook-style checkpoint.",
                ]
                if preprocessing_profile == "kaggle_notebook_v1"
                else [
                    "Convert the uploaded raster to RGB.",
                    f"Resize to {COMMON_IMAGE_SIZE[1]} x {COMMON_IMAGE_SIZE[0]} for common preprocessing.",
                    "Apply a 3 x 3 median filter for denoising.",
                    "Apply CLAHE to the luminance channel.",
                    "Resize to each model's input dimensions.",
                    "Apply each architecture's model-specific pixel preprocessing.",
                ]
            ),
            "skull_background_removal": (
                "Not applied; no validated brain/skull segmentation mask is available."
            ),
        },
        "original_image": _encode_rgb_image(selected_image["image"]),
        "explain": {
            "requested": explain,
            "status": explanation_status,
            "reason": explanation_reason,
            "model": explanation_model_name if explain != "none" else None,
            "class": label if explain != "none" else None,
            "gradcam_overlay": gradcam_overlay,
            "guided_backprop": guided_backprop_image,
            "hybrid_cam_overlay": hybrid_cam_overlay,
            "fusion_method": fusion_method,
            "attribution_overlap": attribution_overlap,
            "slice_index": selected_slice_index,
        },
        "evaluation_metrics": evaluation_metrics,
        "timing_ms": round(sum(elapsed_times.values()), 3),
        "warnings": (
            (["MRI sequence was not specified by the dataset."] if sequence.upper() == "UNSPECIFIED" else [])
            + (
                [f"{explain} unavailable: {explanation_reason}"]
                if explain != "none" and explanation_status != "generated"
                else []
            )
        ),
        "disclaimer": (
            "Research and educational use only; not a clinical diagnostic device. "
            "Dataset class labels are not clinician-confirmed diagnoses."
        ),
    }
