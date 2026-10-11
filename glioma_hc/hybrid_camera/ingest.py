from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pydicom


def load_image_to_8bit(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix in {".dcm", ".dicom"}:
        data = pydicom.dcmread(str(image_path))
        array = data.pixel_array
        if array.dtype != np.uint8:
            if hasattr(data, "RescaleSlope") and hasattr(data, "RescaleIntercept"):
                array = data.pixel_array.astype(np.float32)
                array = array * float(data.RescaleSlope) + float(data.RescaleIntercept)
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(arr, dtype=np.uint8)
    return arr
