from __future__ import annotations

import cv2
import numpy as np


COMMON_IMAGE_SIZE = (300, 300)


def preprocess_mri_rgb(image: np.ndarray) -> np.ndarray:
    """Apply deterministic denoising and luminance CLAHE to an RGB MRI image."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("MRI preprocessing expects an RGB image with three channels.")
    if not np.isfinite(rgb).all():
        raise ValueError("MRI image contains non-finite pixel values.")

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    denoised = cv2.medianBlur(rgb, 3)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    luminance, chroma_a, chroma_b = cv2.split(lab)
    enhanced_luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    enhanced = cv2.merge((enhanced_luminance, chroma_a, chroma_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
