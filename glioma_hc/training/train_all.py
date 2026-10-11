from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from glioma_hc.config import SETTINGS
from glioma_hc.dataset_builder.make_smoke_data import make_smoke_dataset
from glioma_hc.evaluation.evaluate import compute_multiclass_metrics
from glioma_hc.training.ensemble import (
    choose_deployment_candidate,
    choose_ensemble_strategy,
)
from glioma_hc.training.data_loader import build_image_dataset
from glioma_hc.training.model_zoo import (
    MODEL_SPECS,
    REQUIRED_ENSEMBLE_MODELS,
    build_model,
    list_supported_models,
)

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - runtime dependency is optional
    tf = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GliomaScan-HC models")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(REQUIRED_ENSEMBLE_MODELS),
        help="Model names to train (defaults to all eight ensemble members)",
    )
    parser.add_argument("--preset", choices=["paper", "fast"], default="paper")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-dir", type=Path, help="Processed dataset directory containing train/ and val/ class folders")
    parser.add_argument("--model-dir", type=Path, help="Directory for trained model checkpoints")
    parser.add_argument("--fine-tune", action="store_true", help="Train the pretrained backbone as well as the classification head")
    parser.add_argument("--epochs", type=int, help="Override the epoch count (default: 10 for real-data runs)")
    parser.add_argument("--batch-size", type=int, help="Override the training and validation batch size")
    parser.add_argument("--seed", type=int, default=SETTINGS.seed, help="Random seed for data shuffling and model initialization")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Classification-head learning rate")
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5, help="Low learning rate for staged backbone fine-tuning")
    parser.add_argument("--summary-output", type=Path, help="Optional summary path; defaults to reports/training_summary.json")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--no-balance", action="store_false", dest="balance")
    parser.set_defaults(balance=True)
    return parser.parse_args()


def _ensure_directory(project_root: Path) -> Path:
    project_root = Path(project_root)
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "reports").mkdir(parents=True, exist_ok=True)
    return project_root


def _write_summary(
    summary: dict[str, Any],
    project_root: Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    _ensure_directory(project_root)
    summary_path = (
        Path(output_path)
        if output_path is not None
        else project_root / "reports" / "training_summary.json"
    )
    if not summary_path.is_absolute():
        summary_path = project_root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.is_file():
        backup_path = summary_path.with_name(f"{summary_path.stem}_backup{summary_path.suffix}")
        backup_index = 2
        while backup_path.exists():
            backup_path = summary_path.with_name(
                f"{summary_path.stem}_backup{backup_index}{summary_path.suffix}"
            )
            backup_index += 1
        backup_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _dataset_image_counts(data_dir: Path) -> dict[str, dict[str, int]]:
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    counts = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        counts[split] = {
            class_name: (
                sum(
                    path.is_file() and path.suffix.lower() in supported_extensions
                    for path in (split_dir / class_name).iterdir()
                )
                if (split_dir / class_name).is_dir()
                else 0
            )
            for class_name in SETTINGS.label_order
        }
    return counts


def run_training(
    model_names: list[str] | tuple[str, ...] | None = None,
    preset: str = "paper",
    quick: bool = False,
    project_root: str | Path | None = None,
    balance: bool = True,
    data_dir_override: str | Path | None = None,
    model_dir_override: str | Path | None = None,
    fine_tune: bool = False,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    seed: int = SETTINGS.seed,
    learning_rate: float = 1e-3,
    fine_tune_learning_rate: float = 1e-5,
    summary_path_override: str | Path | None = None,
) -> dict[str, Any]:
    use_configured_model_dir = project_root is None
    project_root = Path(project_root) if project_root is not None else SETTINGS.project_root
    project_root = _ensure_directory(project_root)

    requested = list(model_names or REQUIRED_ENSEMBLE_MODELS)
    if len(requested) != len(set(requested)):
        raise ValueError("Each model may be requested only once.")
    if set(requested) == set(REQUIRED_ENSEMBLE_MODELS):
        requested = list(REQUIRED_ENSEMBLE_MODELS)
    supported = list_supported_models()
    for name in requested:
        if name not in supported:
            raise ValueError(f"Unsupported model {name}. Choose from {supported}")

    if quick:
        data_dir = project_root / "data" / "smoke" / "processed"
    elif data_dir_override is not None:
        data_dir = Path(data_dir_override)
    else:
        data_dir = project_root / "data" / "processed"
    if quick:
        make_smoke_dataset(data_dir)
    if model_dir_override is not None:
        model_dir = Path(model_dir_override)
    elif use_configured_model_dir:
        model_dir = SETTINGS.models_dir
    else:
        model_dir = project_root / "models"
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if set(requested) == set(REQUIRED_ENSEMBLE_MODELS):
        training_strategy = (
            "eight_cnn_staged_fine_tuning"
            if fine_tune
            else "eight_cnn_transfer_ensemble"
        )
    elif requested == ["kaggle_cnn"]:
        training_strategy = "kaggle_cnn_from_scratch"
    elif requested == ["attention_cnn"]:
        training_strategy = "attention_cnn_from_scratch"
    elif "attention_cnn" in requested or "kaggle_cnn" in requested:
        training_strategy = "mixed_model_specific"
    elif all(not MODEL_SPECS[name].get("pretrained", False) for name in requested):
        training_strategy = "custom_architecture_from_scratch"
    elif not all(MODEL_SPECS[name].get("pretrained", False) for name in requested):
        training_strategy = "mixed_pretrained_and_custom_architectures"
    elif fine_tune:
        training_strategy = "fine_tuning"
    else:
        training_strategy = "frozen_backbone_feature_extraction"

    if batch_size_override is not None and batch_size_override < 1:
        raise ValueError("batch_size_override must be at least 1")
    if (
        not np.isfinite(learning_rate)
        or not np.isfinite(fine_tune_learning_rate)
        or learning_rate <= 0
        or fine_tune_learning_rate <= 0
    ):
        raise ValueError("Learning rates must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    summary: dict[str, Any] = {
        "status": "skipped",
        "preset": preset,
        "quick": quick,
        "balance": balance,
        "fine_tune": fine_tune,
        "training_strategy": training_strategy,
        "project_root": str(project_root),
        "label_order": list(SETTINGS.label_order),
        "training_data_dir": str(data_dir),
        "models": [],
        "dataset_counts": _dataset_image_counts(data_dir),
        "note": "No training was run because TensorFlow is unavailable in this environment.",
    }
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["dataset"] = manifest.get("dataset")
        summary["dataset_manifest"] = str(manifest_path)
        summary["class_mapping"] = manifest.get("class_mapping")
        summary["class_note"] = manifest.get("class_note")
        summary["modality_note"] = manifest.get("modality_note")

    raw_dir = project_root / "data" / "raw"
    if not quick and data_dir_override is None and (not raw_dir.exists() or not any(path.is_file() for path in raw_dir.rglob("*"))):
        summary["note"] = "Training skipped: add real source data under data/raw before training."
        return _write_summary(summary, project_root)

    if tf is None:
        return _write_summary(summary, project_root, summary_path_override)

    if epochs_override is not None and epochs_override < 1:
        raise ValueError("epochs_override must be at least 1")
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print("Dataset image counts before training:")
    print(json.dumps(summary["dataset_counts"], indent=2))
    epochs = 1 if quick else epochs_override or (
        40 if "kaggle_cnn" in requested else 10 if "attention_cnn" in requested else 10
    )
    batch_size = batch_size_override or (
        8 if quick else 32 if requested == ["kaggle_cnn"] else 8
    )
    training_records: list[dict[str, Any]] = []
    validation_probability_sets = []
    validation_labels = None
    validation_model_metrics = {}

    for name in requested:
        input_size = MODEL_SPECS[name]["input_size"]
        backbone_frozen = bool(MODEL_SPECS[name].get("pretrained", False))
        model = build_model(
            name,
            input_shape=(input_size, input_size, 3),
            num_classes=len(SETTINGS.label_order),
            freeze_backbone=backbone_frozen,
        )
        model_path = model_dir / f"{name}_{preset}_seed{seed}.keras"
        suffix = 2
        while model_path.exists():
            model_path = model_dir / f"{name}_{preset}_seed{seed}_run{suffix}.keras"
            suffix += 1
        train_ds = build_image_dataset(
            data_dir / "train",
            image_size=(input_size, input_size),
            batch_size=batch_size,
            model_name=name,
            augment=True,
            seed=seed,
        )
        val_ds = build_image_dataset(
            data_dir / "val",
            image_size=(input_size, input_size),
            batch_size=batch_size,
            model_name=name,
            shuffle=False,
            seed=seed,
        )

        class_counts = summary["dataset_counts"]["train"]
        if any(count == 0 for count in class_counts.values()):
            raise ValueError(f"Training data must contain images for every class: {class_counts}")
        total_samples = sum(class_counts.values())
        class_weights = {
            class_index: total_samples / (len(class_counts) * class_counts[class_name])
            for class_index, class_name in enumerate(SETTINGS.label_order)
        }
        if not balance:
            class_weights = None

        def compile_for_training(rate: float) -> None:
            optimizer = (
                tf.keras.optimizers.Adam(
                    learning_rate=rate,
                    beta_1=0.869,
                    beta_2=0.995,
                )
                if name == "kaggle_cnn"
                else tf.keras.optimizers.Adam(learning_rate=rate)
            )
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        def make_callbacks() -> list[Any]:
            return [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=2,
                ),
            ]

        epoch_history: list[dict[str, float]] = []
        fine_tuned_layer_count = 0
        can_fine_tune = fine_tune and MODEL_SPECS[name].get("pretrained", False)
        head_epochs = max(1, epochs // 2) if can_fine_tune else epochs
        compile_for_training(learning_rate)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=head_epochs,
            callbacks=make_callbacks(),
            class_weight=class_weights,
            verbose=0,
        )
        epoch_history.extend(
            {key: float(values[index]) for key, values in head_history.history.items()}
            for index in range(len(head_history.history["loss"]))
        )
        best_head_loss = min(head_history.history["val_loss"])

        if can_fine_tune and head_epochs < epochs:
            backbone = next(
                (
                    layer
                    for layer in model.layers
                    if isinstance(layer, tf.keras.Model)
                ),
                None,
            )
            if backbone is None:
                raise ValueError(f"Pretrained model {name} has no nested backbone to fine-tune.")
            backbone.trainable = True
            for layer in backbone.layers:
                layer.trainable = False
            unfreeze_count = max(1, int(len(backbone.layers) * 0.2))
            for layer in backbone.layers[-unfreeze_count:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
                    fine_tuned_layer_count += 1

            with tempfile.TemporaryDirectory(prefix="glioma-hc-best-") as temp_dir:
                head_weights_path = Path(temp_dir) / "head.weights.h5"
                model.save_weights(head_weights_path)
                compile_for_training(fine_tune_learning_rate)
                fine_history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    initial_epoch=head_epochs,
                    epochs=epochs,
                    callbacks=make_callbacks(),
                    class_weight=class_weights,
                    verbose=0,
                )
                epoch_history.extend(
                    {
                        key: float(values[index])
                        for key, values in fine_history.history.items()
                    }
                    for index in range(len(fine_history.history["loss"]))
                )
                if min(fine_history.history["val_loss"]) >= best_head_loss:
                    model.load_weights(head_weights_path)

        model.save(model_path)
        checkpoint_validation_accuracy = float(model.evaluate(val_ds, verbose=0)[1])
        model_validation_probabilities = np.asarray(model.predict(val_ds, verbose=0), dtype=np.float64)
        current_validation_labels = np.concatenate(
            [labels.numpy().astype(int) for _, labels in val_ds]
        )
        if validation_labels is not None and not np.array_equal(
            validation_labels, current_validation_labels
        ):
            raise ValueError("Validation class order changed between ensemble members.")
        validation_labels = current_validation_labels
        validation_probability_sets.append(model_validation_probabilities)
        validation_metrics = compute_multiclass_metrics(
            current_validation_labels,
            model_validation_probabilities,
            SETTINGS.label_order,
        )
        validation_model_metrics[name] = validation_metrics
        if name == "attention_cnn":
            model_training_strategy = "attention_cnn_from_scratch"
        elif name == "kaggle_cnn":
            model_training_strategy = "kaggle_cnn_from_scratch"
        elif MODEL_SPECS[name].get("pretrained") and fine_tuned_layer_count > 0:
            model_training_strategy = "fine_tuning"
        elif MODEL_SPECS[name].get("pretrained"):
            model_training_strategy = "frozen_backbone_feature_extraction"
        else:
            model_training_strategy = "custom_architecture_from_scratch"
        record = {
            "name": name,
            "status": "trained",
            "model_path": str(model_path),
            "epochs": len(epoch_history),
            "backbone_frozen": bool(
                MODEL_SPECS[name].get("pretrained", False)
                and fine_tuned_layer_count == 0
            ),
            "training_strategy": (
                "frozen_backbone_feature_extraction"
                if can_fine_tune and fine_tuned_layer_count == 0
                else model_training_strategy
            ),
            "head_training_epochs": head_epochs,
            "fine_tuning_epochs": max(0, len(epoch_history) - head_epochs),
            "fine_tuned_backbone_layers": fine_tuned_layer_count,
            "final_train_accuracy": float(epoch_history[-1]["accuracy"]),
            "final_val_accuracy": checkpoint_validation_accuracy,
            "val_loss": float(min(epoch["val_loss"] for epoch in epoch_history)),
            "epochs_history": [
                {
                    "epoch": epoch + 1,
                    "train_loss": metrics["loss"],
                    "val_loss": metrics["val_loss"],
                    "train_accuracy": metrics["accuracy"],
                    "val_accuracy": metrics["val_accuracy"],
                }
                for epoch, metrics in enumerate(epoch_history)
            ],
            "validation_metrics": validation_metrics,
            "seed": seed,
            "data_augmentation": [
                "random_rotation",
                "small_translation",
                "brightness",
            ],
            "pretrained_imagenet": bool(MODEL_SPECS[name].get("pretrained", False)),
            "class_weights": class_weights,
        }
        training_records.append(record)

    summary.update({
        "status": "ok",
        "models": training_records,
        "note": (
            "Training completed using generated synthetic smoke data; these metrics are only for pipeline validation."
            if quick
            else f"Training completed using {data_dir}; metrics describe this local dataset and are not paper or clinical results."
        ),
    })
    if set(requested) == set(REQUIRED_ENSEMBLE_MODELS):
        model_weights = [record["final_val_accuracy"] for record in training_records]
        ensemble = choose_ensemble_strategy(
            validation_labels,
            np.stack(validation_probability_sets),
            model_weights,
        )
        ensemble["model_order"] = [record["name"] for record in training_records]
        ensemble["model_weights"] = {
            record["name"]: float(model_weights[index] / sum(model_weights))
            for index, record in enumerate(training_records)
        }
        summary["ensemble"] = ensemble
        summary["deployment"] = choose_deployment_candidate(
            validation_model_metrics,
            ensemble,
        )
        summary["seed"] = seed
        summary["batch_size"] = batch_size
        summary["learning_rate"] = learning_rate
        summary["fine_tune_learning_rate"] = fine_tune_learning_rate
    else:
        summary["ensemble"] = {
            "status": "incomplete",
            "required_models": list(REQUIRED_ENSEMBLE_MODELS),
            "trained_models": [record["name"] for record in training_records],
            "missing_models": [
                name for name in REQUIRED_ENSEMBLE_MODELS if name not in requested
            ],
            "selection_split": "val",
        }
    return _write_summary(summary, project_root, summary_path_override)


def main() -> None:
    args = parse_args()
    summary = run_training(
        model_names=args.models,
        preset=args.preset,
        quick=args.quick,
        project_root=Path(__file__).resolve().parents[2],
        balance=args.balance,
        data_dir_override=args.data_dir,
        fine_tune=args.fine_tune,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        summary_path_override=args.summary_output,
        model_dir_override=args.model_dir or SETTINGS.models_dir,
    )

    if summary["status"] == "ok":
        print(json.dumps({"status": "ok", "models": summary["models"]}, indent=2))
    else:
        print(json.dumps({"status": "skipped", "note": summary["note"]}, indent=2))


if __name__ == "__main__":
    main()
