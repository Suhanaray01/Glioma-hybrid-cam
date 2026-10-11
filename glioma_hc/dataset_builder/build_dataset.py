from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from glioma_hc.config import SETTINGS


def validate_label_order(labels: Iterable[str]) -> list[str]:
    ordered = list(labels)
    if ordered != SETTINGS.label_order:
        raise ValueError(f"Label order must be {SETTINGS.label_order}, got {ordered}")
    return ordered


def paper_split_indices(labels: Iterable[int], seed: int = 42, test_fraction: float = 0.2):
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    classes = sorted(set(labels))
    split = {cls: [] for cls in classes}
    for idx in indices:
        split[labels[idx]].append(idx)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in classes:
        arr = split[cls]
        n_test = max(1, round(len(arr) * test_fraction))
        if n_test >= len(arr):
            n_test = max(1, len(arr) - 1)
        test_idx.extend(arr[:n_test])
        train_idx.extend(arr[n_test:])
    return train_idx, test_idx


def label_ordered_counts(image_dir: Path):
    counts = {}
    for label in SETTINGS.label_order:
        folder = image_dir / label
        counts[label] = len(list(folder.glob("*"))) if folder.exists() else 0
    return counts


def augment_train_only(train_dir: Path, class_name: str, target_count: int, seed: int = 42) -> dict[str, int]:
    """A lightweight augmentation policy for the normal class in training-only files.
    Real augmentation should create rotated/reflected/noise/blur copies while keeping the
    validation split untouched.
    """
    rng = random.Random(seed)
    files = sorted((train_dir / class_name).glob("*"))
    if not files:
        return {"original": 0, "augmented": 0, "final": 0}

    if class_name == "normal":
        created = 0
        for idx, original in enumerate(files):
            if created >= target_count - len(files):
                break
            duplicate = train_dir / class_name / f"{original.stem}_aug_{idx}.png"
            duplicate.write_bytes(original.read_bytes())
            created += 1
        return {"original": len(files), "augmented": created, "final": len(files) + created}

    return {"original": len(files), "augmented": 0, "final": len(files)}


def build_synthetic_paper_split(root: str | Path = SETTINGS.processed_dir):
    root_path = Path(root)
    per_class = {label: 20 for label in SETTINGS.label_order}
    train = []
    test = []
    for label in SETTINGS.label_order:
        label_dir = root_path / "train" / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(per_class[label]):
            train.append((label_dir / f"{label}_{idx}.png", label))
        test_dir = root_path / "test" / label
        test_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(5):
            test.append((test_dir / f"{label}_test_{idx}.png", label))
    return {"train": len(train), "test": len(test), "counts": {label: per_class[label] for label in SETTINGS.label_order}}


def build_dataset_summary(root: str | Path = SETTINGS.processed_dir):
    root_path = Path(root)
    counts = {}
    for split in ["train", "val", "test"]:
        counts[split] = label_ordered_counts(root_path / split)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic dataset layout that matches the expected paper-style split.")
    parser.add_argument("--root", type=str, default=str(SETTINGS.processed_dir), help="Output directory for processed images")
    args = parser.parse_args()
    summary = build_synthetic_paper_split(args.root)
    print(summary)


if __name__ == "__main__":
    main()
