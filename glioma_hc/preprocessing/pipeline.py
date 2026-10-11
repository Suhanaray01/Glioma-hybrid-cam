from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def preprocess_image(image: np.ndarray, disk_radius: int = 12, clip_limit: float = 2.0) -> dict[str, Any]:
    gray = ensure_grayscale(image)
    kernel = morphology.disk(disk_radius)
    opened = morphology.opening(gray, kernel)
    difference = gray - opened
    diff_uint8 = np.clip(difference, 0, 255).astype(np.uint8)
    thresh_value = max(10.0, float(threshold_otsu(diff_uint8)))
    mask = difference > thresh_value

    if np.any(mask):
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                idx = int(np.argmax(areas)) + 1
                skull_mask = (labels == idx).astype(np.uint8)
            else:
                skull_mask = mask.astype(np.uint8)
        else:
            skull_mask = mask.astype(np.uint8)
    else:
        skull_mask = np.zeros_like(gray, dtype=np.uint8)

    brain = np.clip(gray - (skull_mask * 255.0), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    equalized = clahe.apply(brain)
    rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    return {
        "original": gray.astype(np.uint8),
        "opened": opened.astype(np.uint8),
        "skull_mask": skull_mask.astype(np.uint8),
        "brain": brain,
        "clahe": equalized,
        "rgb": rgb,
    }


def skull_removal_pipeline(image: np.ndarray, disk_radius: int = 12, clip_limit: float = 2.0) -> dict[str, Any]:
    return preprocess_image(image, disk_radius=disk_radius, clip_limit=clip_limit)
