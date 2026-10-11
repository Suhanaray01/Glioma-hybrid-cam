from __future__ import annotations

import cv2
import numpy as np


def camera_simulation(image: np.ndarray, severity: float = 0.5) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    image_float = arr.astype(np.float32)
    brightness = 1.0 + severity * 0.35
    image_float = np.clip(image_float * brightness, 0, 255)

    if severity > 0.15:
        blur_kernel = max(3, int(3 + severity * 9))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        image_float = cv2.GaussianBlur(image_float, (blur_kernel, blur_kernel), 0)

    if severity > 0.3:
        noise = np.random.normal(0, severity * 16, image_float.shape).astype(np.float32)
        image_float = np.clip(image_float + noise, 0, 255)

    return image_float.astype(np.uint8)
