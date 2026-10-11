from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from glioma_hc.config import SETTINGS


def make_smoke_dataset(output_dir: str | Path | None = None, count_per_class: int = 8) -> Path:
    root = Path(output_dir) if output_dir is not None else SETTINGS.processed_dir
    for split in ["train", "val", "test"]:
        for label in SETTINGS.label_order:
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for idx in range(count_per_class):
                arr = np.zeros((128, 128, 3), dtype=np.uint8)
                class_index = SETTINGS.label_order.index(label)
                arr[:, :, class_index % 3] = 60 + class_index * 35 + idx
                arr[:, :, (class_index + 1) % 3] = 20 + idx
                image_path = folder / f"{label}_{split}_{idx}.png"
                import cv2

                cv2.imwrite(str(image_path), arr)
    return root


if __name__ == "__main__":
    make_smoke_dataset()
    print("Synthetic smoke dataset created.")
