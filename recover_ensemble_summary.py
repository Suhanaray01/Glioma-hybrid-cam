
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glioma_hc.config import SETTINGS
from glioma_hc.evaluation.evaluate import compute_multiclass_metrics
from glioma_hc.inference import _load_keras_model
from glioma_hc.training.data_loader import build_image_dataset
from glioma_hc.training.ensemble import (
    choose_deployment_candidate,
    choose_ensemble_strategy,
)
from glioma_hc.training.model_zoo import (
    MODEL_SPECS,
    REQUIRED_ENSEMBLE_MODELS,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = (ROOT / "saved_models").resolve()
SUMMARY_PATH = ROOT / "reports" / "training_summary.json"
DATA_DIR = ROOT / "data" / "processed"

# Preserve the current summary as a backup.
if SUMMARY_PATH.exists():
    backup = SUMMARY_PATH.with_name("training_summary_before_recovery.json")
    if not backup.exists():
        backup.write_text(
            SUMMARY_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

records = []
prediction_sets = []
true_labels = None
model_metrics = {}

print("Recovering ensemble summary from saved models...")
print("Validation data:", DATA_DIR / "val")

for name in REQUIRED_ENSEMBLE_MODELS:
    model_path = MODEL_DIR / f"{name}_fast.keras"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing saved model: {model_path}"
        )

    print(f"\nLoading {name}...")
    model = _load_keras_model(model_path)

    size = MODEL_SPECS[name]["input_size"]
    dataset = build_image_dataset(
        DATA_DIR / "val",
        image_size=(size, size),
        model_name=name,
        shuffle=False,
    )

    probabilities = np.asarray(
        model.predict(dataset, verbose=0),
        dtype=np.float64,
    )
    labels = np.concatenate(
        [y.numpy().astype(int) for _, y in dataset]
    )

    if probabilities.shape != (
        len(labels),
        len(SETTINGS.label_order),
    ):
        raise ValueError(
            f"Unexpected output shape for {name}: "
            f"{probabilities.shape}"
        )

    if true_labels is not None and not np.array_equal(
        true_labels, labels
    ):
        raise ValueError(
            "Validation labels differ between models."
        )

    true_labels = labels
    prediction_sets.append(probabilities)

    metrics = compute_multiclass_metrics(
        labels,
        probabilities,
        SETTINGS.label_order,
    )
    model_metrics[name] = metrics

    accuracy = float(np.mean(
        np.argmax(probabilities, axis=1) == labels
    ))

    records.append({
        "name": name,
        "status": "trained",
        "model_path": str(model_path),
        "final_val_accuracy": accuracy,
        "final_train_accuracy": None,
        "epochs": 1,
        "training_strategy": "recovered_existing_checkpoint",
        "backbone_frozen": None,
        "pretrained_imagenet": bool(
            MODEL_SPECS[name].get("pretrained", False)
        ),
        "validation_metrics": metrics,
        "note": (
            "Recovered from an existing one-epoch checkpoint. "
            "Original training history was not reconstructed."
        ),
    })

    print(f"Validation accuracy: {accuracy:.2%}")

print("\nSelecting voting strategy from validation predictions...")
weights = [r["final_val_accuracy"] for r in records]

selection = choose_ensemble_strategy(
    true_labels,
    np.stack(prediction_sets),
    weights,
)

normalized_weights = np.asarray(weights, dtype=np.float64)
if normalized_weights.sum() > 0:
    normalized_weights /= normalized_weights.sum()
else:
    normalized_weights = np.full(
        len(weights), 1.0 / len(weights)
    )

selection["model_order"] = list(REQUIRED_ENSEMBLE_MODELS)
selection["model_weights"] = {
    name: float(normalized_weights[i])
    for i, name in enumerate(REQUIRED_ENSEMBLE_MODELS)
}

summary = {
    "status": "ok",
    "preset": "fast",
    "quick": False,
    "balance": True,
    "fine_tune": False,
    "training_strategy": "recovered_existing_checkpoints",
    "project_root": str(ROOT),
    "label_order": list(SETTINGS.label_order),
    "training_data_dir": str(DATA_DIR),
    "models": records,
    "ensemble": selection,
    "deployment": choose_deployment_candidate(model_metrics, selection),
    "dataset_counts": {},
    "note": (
        "Summary reconstructed from existing checkpoints. "
        "Models were trained separately for one epoch each. "
        "Original training histories and train accuracies "
        "are unavailable in this reconstructed summary."
    ),
}

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(
    json.dumps(summary, indent=2, default=str),
    encoding="utf-8",
)

print("\nSummary saved to:", SUMMARY_PATH)
print("Selected strategy:", selection["strategy"])
print("Validation comparisons:")
print(json.dumps(selection.get("comparisons", {}), indent=2))
print("\nRecovery complete. No models were retrained.")
