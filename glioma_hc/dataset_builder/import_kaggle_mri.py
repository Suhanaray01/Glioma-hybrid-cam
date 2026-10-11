from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

from glioma_hc.config import SETTINGS


DATASET_ID = "masoudnickparvar/brain-tumor-mri-dataset"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_MAPPING = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "pituitary": "pituitary",
}
CLASS_ORDER = tuple(SETTINGS.label_order)


def _find_child(directory: Path, name: str) -> Path:
    for child in directory.iterdir():
        if child.is_dir() and child.name.casefold() == name.casefold():
            return child
    raise FileNotFoundError(f"Expected {name!r} folder under {directory}")


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def _class_files(split_dir: Path) -> dict[str, list[Path]]:
    result = {label: [] for label in CLASS_ORDER}
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        source_class = class_dir.name.casefold().replace(" ", "_")
        target_class = CLASS_MAPPING.get(source_class)
        if target_class is None:
            raise ValueError(f"Unsupported class folder {class_dir.name!r} in {split_dir}")
        result[target_class].extend(_image_files(class_dir))
    return result


def _copy_images(images: Iterable[Path], target_dir: Path, label: str) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for index, source in enumerate(images):
        destination = target_dir / f"{label}_{index:05d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied += 1
    return copied


def import_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> dict[str, object]:
    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    training_dir = _find_child(source_root, "Training")
    testing_dir = _find_child(source_root, "Testing")
    source_training = _class_files(training_dir)
    source_testing = _class_files(testing_dir)
    for label in CLASS_ORDER:
        if not source_training[label] or not source_testing[label]:
            raise ValueError(f"Missing mapped class {label!r} in Training or Testing")

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for label in CLASS_ORDER:
        training_images = source_training[label].copy()
        rng.shuffle(training_images)
        validation_count = max(1, round(len(training_images) * validation_fraction))
        if validation_count >= len(training_images):
            raise ValueError(f"Not enough {label} training images to create a validation split")
        validation_images = training_images[:validation_count]
        fit_images = training_images[validation_count:]
        counts["train"][label] = _copy_images(fit_images, output_root / "train" / label, label)
        counts["val"][label] = _copy_images(validation_images, output_root / "val" / label, label)
        counts["test"][label] = _copy_images(source_testing[label], output_root / "test" / label, label)

    manifest = {
        "dataset": DATASET_ID,
        "source_path": str(source_root),
        "license": "CC BY 4.0",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "label_order": list(CLASS_ORDER),
        "class_mapping": {key: value for key, value in CLASS_MAPPING.items() if key != "no_tumor"},
        "class_note": "Dataset class labels are source categories, not clinician-confirmed diagnoses.",
        "modality_note": "The dataset does not specify MRI sequences; results do not establish T2-FLAIR-specific performance.",
        "counts": counts,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all four Kaggle brain MRI classes for training.")
    parser.add_argument("--source", required=True, help="Path printed by kagglehub.dataset_download")
    parser.add_argument("--output", required=True, help="Empty destination directory for processed splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = import_dataset(args.source, args.output, seed=args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()