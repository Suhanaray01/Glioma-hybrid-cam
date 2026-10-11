from __future__ import annotations

import cv2
import numpy as np


def assess_capture_quality(image: np.ndarray) -> dict:
    arr = np.asarray(image)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr.astype(np.uint8)

    sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    brightness = float(np.mean(gray))
    glare_pct = float(np.mean(gray > 245) * 100.0)
    roi_found = bool(gray.size > 0 and brightness > 10)
    ok = bool(roi_found and sharpness > 20.0 and glare_pct < 75.0 and brightness < 245)

    tips: list[str] = []
    if sharpness <= 20:
        tips.append("Image is blurry; reduce motion and increase focus.")
    if brightness < 30:
        tips.append("Image is too dark; increase lighting.")
    if brightness > 230:
        tips.append("Image is overexposed; reduce glare or exposure.")
    if glare_pct > 50:
        tips.append("Glare is high; avoid reflections or tilt the image.")
    if not roi_found:
        tips.append("No useful region detected; ensure the MRI fits within the frame.")
    if not tips:
        tips.append("Capture quality looks acceptable for preprocessing.")

    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "glare_pct": round(glare_pct, 2),
        "roi_found": roi_found,
        "ok": ok,
        "tips": tips,
    }
