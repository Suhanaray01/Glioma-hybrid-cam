from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from glioma_hc.config import SETTINGS


def compute_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "sensitivity": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(
            (np.sum((y_true_arr == 0) & (y_pred == 0)) / max(np.sum(y_true_arr == 0), 1))
        ),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true_arr, y_pred)),
    }

    if len(np.unique(y_true_arr)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        metrics["auc"] = 0.0

    return metrics


def compute_multiclass_metrics(
    y_true: list[int],
    probabilities: list[list[float]] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    probability_arr = np.asarray(probabilities, dtype=float)
    class_indices = list(range(len(class_names)))
    if y_true_arr.ndim != 1 or y_true_arr.size == 0:
        raise ValueError("Evaluation labels must be a non-empty one-dimensional array.")
    if probability_arr.shape != (len(y_true_arr), len(class_names)):
        raise ValueError("Prediction probabilities must have one column per configured class.")
    if np.any(y_true_arr < 0) or np.any(y_true_arr >= len(class_names)):
        raise ValueError("Evaluation labels are outside the configured class order.")
    if (
        not np.isfinite(probability_arr).all()
        or np.any(probability_arr < 0)
        or np.any(probability_arr > 1)
        or not np.allclose(probability_arr.sum(axis=1), 1.0, atol=1e-3)
    ):
        raise ValueError("Prediction outputs must be finite normalized probabilities.")

    predictions = np.argmax(probability_arr, axis=1)
    per_class_precision = precision_score(
        y_true_arr, predictions, labels=class_indices, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true_arr, predictions, labels=class_indices, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true_arr, predictions, labels=class_indices, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true_arr, predictions, labels=class_indices)
    supports = np.bincount(y_true_arr, minlength=len(class_names))
    true_positives = np.diag(matrix)
    false_negatives = supports - true_positives
    false_positives = np.bincount(predictions, minlength=len(class_names)) - true_positives
    true_negatives = len(y_true_arr) - true_positives - false_negatives - false_positives
    per_class_specificity = true_negatives / np.maximum(
        true_negatives + false_positives, 1
    )
    per_class_auc = []
    for class_index in class_indices:
        binary_labels = (y_true_arr == class_index).astype(int)
        if len(np.unique(binary_labels)) == 2:
            auc = float(roc_auc_score(binary_labels, probability_arr[:, class_index]))
            per_class_auc.append(auc)
        else:
            per_class_auc.append(None)

    supported_auc = [auc for auc in per_class_auc if auc is not None]
    return {
        "accuracy": float(accuracy_score(y_true_arr, predictions)),
        "macro_precision": float(np.mean(per_class_precision)),
        "macro_recall": float(np.mean(per_class_recall)),
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "macro_specificity": float(np.mean(per_class_specificity)),
        "macro_f1": float(np.mean(per_class_f1)),
        "macro_auc": float(np.mean(supported_auc)) if supported_auc else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y_true_arr, predictions)),
        "per_class": {
            class_name: {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "sensitivity": float(per_class_recall[index]),
                "specificity": float(per_class_specificity[index]),
                "f1": float(per_class_f1[index]),
                "auc": per_class_auc[index],
                "support": int(supports[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def write_summary(metrics: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        backup = output.with_name(f"{output.stem}_backup{output.suffix}")
        backup_index = 2
        while backup.exists():
            backup = output.with_name(
                f"{output.stem}_backup{backup_index}{output.suffix}"
            )
            backup_index += 1
        backup.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    with output.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def evaluate_model(
    model_path: str | Path,
    data_dir: str | Path,
    model_name: str,
    output_path: str | Path,
    split: str = "test",
    preprocessing_profile: str = "shared_mri",
) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for model evaluation.") from exc
    from glioma_hc.inference import _load_keras_model
    from glioma_hc.training.data_loader import build_image_dataset
    from glioma_hc.training.model_zoo import MODEL_SPECS

    model_file = Path(model_path).resolve()
    if split not in {"test", "val"}:
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    evaluation_dir = Path(data_dir).resolve() / split
    if not model_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation split not found: {evaluation_dir}")

    dataset = build_image_dataset(
        evaluation_dir,
        image_size=(MODEL_SPECS[model_name]["input_size"],) * 2,
        model_name=model_name,
        shuffle=False,
        preprocessing_profile=preprocessing_profile,
    )
    model = _load_keras_model(model_file)
    true_labels = []
    class_probabilities = []
    for images, labels in dataset:
        probabilities = np.asarray(model.predict(images, verbose=0), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(SETTINGS.label_order):
            raise ValueError(f"Model returned invalid prediction shape: {probabilities.shape}")
        true_labels.extend(labels.numpy().astype(int).tolist())
        class_probabilities.extend(probabilities.tolist())

    metrics = compute_multiclass_metrics(true_labels, class_probabilities, SETTINGS.label_order)
    manifest_path = Path(data_dir) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    split_counts = manifest.get("counts", {})
    report = {
        "model": model_file.name,
        "model_path": str(model_file),
        "dataset": manifest.get("dataset", "unspecified"),
        "evaluation_split": split,
        "evaluation_path": str(evaluation_dir),
        "test_split": str(evaluation_dir) if split == "test" else None,
        "test_samples": len(true_labels) if split == "test" else None,
        "evaluation_samples": len(true_labels),
        "dataset_splits": split_counts,
        "dataset_license": manifest.get("license"),
        "label_order": SETTINGS.label_order,
        "class_mapping": manifest.get("class_mapping"),
        "class_note": manifest.get("class_note"),
        "modality_note": manifest.get("modality_note"),
        "metadata": {
            "preprocessing_profile": preprocessing_profile,
            "checkpoint_size_bytes": model_file.stat().st_size,
            "checkpoint_modified_time_ns": model_file.stat().st_mtime_ns,
        },
        "metrics": metrics,
    }
    write_summary(report, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for a smoke or real classification run.")
    parser.add_argument("--model", type=Path, help="Saved Keras checkpoint to evaluate on a held-out test split")
    parser.add_argument("--data-dir", type=Path, help="Processed dataset directory containing val/ and test/")
    parser.add_argument("--model-name", default="kaggle_cnn", help="Model name used for preprocessing")
    parser.add_argument(
        "--preprocessing-profile",
        choices=["shared_mri", "kaggle_notebook_v1"],
        default="shared_mri",
        help="Input preprocessing profile used by the saved checkpoint",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluate on validation or held-out test split")
    parser.add_argument("--true", nargs="*", help="Ground-truth labels in [0, 1] format")
    parser.add_argument("--prob", nargs="*", help="Predicted positive probabilities")
    parser.add_argument("--output", type=str, default="reports/results.json", help="Where to save the report")
    args = parser.parse_args()

    if args.model is not None:
        if args.data_dir is None:
            parser.error("--data-dir is required when --model is provided")
        report = evaluate_model(
            args.model,
            args.data_dir,
            args.model_name,
            args.output,
            split=args.split,
            preprocessing_profile=args.preprocessing_profile,
        )
    else:
        if args.true is None or args.prob is None:
            parser.error("provide --model and --data-dir for real evaluation, or both --true and --prob for explicit metric inputs")
        y_true = [int(value) for value in args.true]
        y_prob = [float(value) for value in args.prob]
        report = {"metrics": compute_metrics(y_true, y_prob)}
        write_summary(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
