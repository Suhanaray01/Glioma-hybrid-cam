from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from glioma_hc.config import SETTINGS
from glioma_hc.evaluation.artifacts import (
    relative_figure_paths,
    save_evaluation_figures,
    summarize_prediction_confidence,
)
from glioma_hc.evaluation.evaluate import compute_multiclass_metrics, write_summary
from glioma_hc.training.data_loader import build_image_dataset
from glioma_hc.training.ensemble import (
    choose_deployment_candidate,
    choose_ensemble_strategy,
    combine_model_probabilities,
    deployment_from_training_summary,
)
from glioma_hc.training.model_zoo import MODEL_SPECS, REQUIRED_ENSEMBLE_MODELS


def evaluate_ensemble(
    data_dir: str | Path,
    training_summary_path: str | Path,
    output_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    try:
        summary = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The eight-model training summary is missing or invalid.") from exc
    if summary.get("status") != "ok" or summary.get("quick") is not False:
        raise ValueError("Evaluation requires completed non-smoke training.")
    if summary.get("label_order") != SETTINGS.label_order:
        raise ValueError("Training summary class order does not match the configured class order.")

    records_by_name = {
        record.get("name"): record
        for record in summary.get("models", [])
        if isinstance(record, dict)
    }
    missing = [name for name in REQUIRED_ENSEMBLE_MODELS if name not in records_by_name]
    if missing:
        raise ValueError("Cannot evaluate incomplete eight-model ensemble: " + ", ".join(missing))

    from glioma_hc.inference import _load_keras_model

    prediction_sets = []
    true_labels = None
    model_metrics = {}
    model_weights = []
    for name in REQUIRED_ENSEMBLE_MODELS:
        record = records_by_name[name]
        model_path = Path(record.get("model_path", "")).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found for {name}: {model_path}")
        model = _load_keras_model(model_path)
        input_size = MODEL_SPECS[name]["input_size"]
        dataset = build_image_dataset(
            Path(data_dir) / split,
            image_size=(input_size, input_size),
            model_name=name,
            shuffle=False,
        )
        predictions = np.asarray(model.predict(dataset, verbose=0), dtype=np.float64)
        labels = np.concatenate([batch_labels.numpy().astype(int) for _, batch_labels in dataset])
        if predictions.shape != (len(labels), len(SETTINGS.label_order)):
            raise ValueError(f"{name} returned invalid prediction dimensions: {predictions.shape}")
        if true_labels is not None and not np.array_equal(true_labels, labels):
            raise ValueError("Dataset label ordering differed between ensemble models.")
        true_labels = labels
        prediction_sets.append(predictions)
        model_metrics[name] = compute_multiclass_metrics(
            labels,
            predictions,
            SETTINGS.label_order,
        )
        model_weights.append(float(record.get("final_val_accuracy", 0.0)))

    model_probability_tensor = np.stack(prediction_sets)
    if split == "val":
        selection = choose_ensemble_strategy(
            true_labels,
            model_probability_tensor,
            model_weights,
        )
        strategy = selection["strategy"]
        normalized_weights = np.asarray(model_weights, dtype=np.float64)
        if normalized_weights.sum() <= 0:
            raise ValueError("Validation-derived model weights must have a positive sum.")
        normalized_weights /= normalized_weights.sum()
        validation_comparison = selection["comparisons"]
        validation_model_metrics = model_metrics
    else:
        ensemble_config = summary.get("ensemble", {})
        if ensemble_config.get("selection_split") != "val":
            raise ValueError("The ensemble voting strategy must be selected on validation data first.")
        strategy = ensemble_config.get("strategy")
        if strategy not in {"hard_voting", "soft_voting", "weighted_soft_voting"}:
            raise ValueError("The training summary does not contain a validation-selected strategy.")
        normalized_weights = np.asarray(
            [
                ensemble_config.get("model_weights", {}).get(name, 0.0)
                for name in REQUIRED_ENSEMBLE_MODELS
            ],
            dtype=np.float64,
        )
        if not np.isfinite(normalized_weights).all() or np.any(normalized_weights < 0):
            raise ValueError("Validation-derived model weights are invalid.")
        if normalized_weights.sum() <= 0:
            raise ValueError("Validation-derived model weights must have a positive sum.")
        normalized_weights /= normalized_weights.sum()
        if strategy == "weighted_soft_voting" and (
            not np.isfinite(normalized_weights).all() or normalized_weights.sum() <= 0
        ):
            raise ValueError("Validation-derived model weights are missing or invalid.")
        validation_comparison = ensemble_config.get("comparisons", {})
        validation_model_metrics = {
            name: records_by_name[name].get("validation_metrics", {})
            for name in REQUIRED_ENSEMBLE_MODELS
        }

    combined_probabilities = combine_model_probabilities(
        model_probability_tensor,
        strategy,
        normalized_weights if strategy == "weighted_soft_voting" else None,
    )
    test_strategy_comparison = {
        candidate: compute_multiclass_metrics(
            true_labels,
            combine_model_probabilities(
                model_probability_tensor,
                candidate,
                normalized_weights if candidate == "weighted_soft_voting" else None,
            ),
            SETTINGS.label_order,
        )
        for candidate in ("hard_voting", "soft_voting", "weighted_soft_voting")
    }
    manifest_path = Path(data_dir) / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Dataset manifest is invalid.") from exc

    deployment = deployment_from_training_summary(summary)
    if deployment is None and split == "val":
        deployment = choose_deployment_candidate(
            validation_model_metrics,
            {
                "comparisons": validation_comparison,
            },
        )
    if deployment is None:
        raise ValueError("The training summary has no validation-selected deployment candidate.")
    if deployment.get("selection_split") != "val":
        raise ValueError("The deployment candidate must be selected using validation data.")
    if deployment.get("kind") == "single_model":
        selected_name = deployment.get("model")
        if selected_name not in REQUIRED_ENSEMBLE_MODELS:
            raise ValueError("The selected deployment model is not an ensemble member.")
        selected_probabilities = prediction_sets[REQUIRED_ENSEMBLE_MODELS.index(selected_name)]
    elif deployment.get("kind") == "ensemble":
        selected_name = None
        selected_probabilities = combined_probabilities
    else:
        raise ValueError("The validation-selected deployment candidate is invalid.")

    selected_metrics = compute_multiclass_metrics(
        true_labels,
        selected_probabilities,
        SETTINGS.label_order,
    )
    deployment_report = {
        **deployment,
        "test_metrics" if split == "test" else "evaluation_metrics": selected_metrics,
    }
    report_path = Path(output_path)
    figures = save_evaluation_figures(
        true_labels,
        selected_probabilities,
        SETTINGS.label_order,
        report_path.parent / "figures",
        report_path.stem,
    )

    report = {
        "model": selected_name or "Eight-CNN ensemble",
        "models": list(REQUIRED_ENSEMBLE_MODELS),
        "model_checkpoint_paths": {
            name: str(Path(records_by_name[name]["model_path"]).resolve())
            for name in REQUIRED_ENSEMBLE_MODELS
        },
        "dataset": manifest.get("dataset", "unspecified"),
        "evaluation_split": split,
        "evaluation_samples": int(len(true_labels)),
        "test_samples": int(len(true_labels)) if split == "test" else None,
        "dataset_splits": manifest.get("counts", {}),
        "dataset_license": manifest.get("license"),
        "label_order": list(SETTINGS.label_order),
        "class_mapping": manifest.get("class_mapping"),
        "class_note": manifest.get("class_note"),
        "modality_note": manifest.get("modality_note"),
        "ensemble": {
            "strategy": strategy,
            "selection_split": "val",
            "model_order": list(REQUIRED_ENSEMBLE_MODELS),
            "model_weights": {
                name: float(normalized_weights[index])
                for index, name in enumerate(REQUIRED_ENSEMBLE_MODELS)
            },
            "validation_comparison": validation_comparison,
            "exploratory_test_strategy_comparison": (
                test_strategy_comparison if split == "test" else None
            ),
        },
        "per_model_metrics": model_metrics,
        "ensemble_metrics": compute_multiclass_metrics(
            true_labels,
            combined_probabilities,
            SETTINGS.label_order,
        ),
        "deployment": deployment_report,
        "metrics": selected_metrics,
        "prediction_confidence": summarize_prediction_confidence(selected_probabilities),
        "figures": relative_figure_paths(figures, report_path.parent),
        "metadata": {
            "selection_split": "val",
            "evaluation_split": split,
            "test_used_for_selection": False,
            "comparison_metrics_are_exploratory": split == "test",
            "seed": summary.get("seed"),
            "training_strategy": summary.get("training_strategy"),
        },
    }
    write_summary(report, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all eight CNNs and their validation-selected ensemble."
    )
    parser.add_argument("--data-dir", type=Path, default=SETTINGS.processed_dir)
    parser.add_argument("--training-summary", type=Path, default=SETTINGS.reports_dir / "training_summary.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or SETTINGS.reports_dir / (
        "ensemble_validation.json" if args.split == "val" else "final_test_results.json"
    )
    report = evaluate_ensemble(args.data_dir, args.training_summary, output, args.split)
    print(json.dumps({
        "evaluation_split": report["evaluation_split"],
        "strategy": report["ensemble"]["strategy"],
        "accuracy": report["metrics"]["accuracy"],
        "macro_specificity": report["metrics"]["macro_specificity"],
        "cohen_kappa": report["metrics"]["cohen_kappa"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
