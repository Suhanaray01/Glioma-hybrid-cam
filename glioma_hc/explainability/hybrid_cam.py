from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from glioma_hc.explainability.gradcam import encode_overlay, make_gradcam_heatmap
from glioma_hc.explainability.guided_backprop import make_guided_backprop


class HybridCAMUnavailable(RuntimeError):
    pass


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values)


def fuse_heatmaps(
    gradcam_heatmap: np.ndarray,
    guided_backprop_map: np.ndarray,
) -> tuple[np.ndarray, float]:
    gradcam = np.asarray(gradcam_heatmap, dtype=np.float32)
    guided = np.asarray(guided_backprop_map, dtype=np.float32)
    if gradcam.ndim != 2 or guided.ndim != 2:
        raise ValueError("Grad-CAM and Guided Backpropagation inputs must be 2D maps.")

    guided = cv2.resize(
        normalize_heatmap(guided),
        (gradcam.shape[1], gradcam.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    gradcam = normalize_heatmap(gradcam)
    combined = normalize_heatmap(gradcam * guided)

    gradcam_support = gradcam >= np.quantile(gradcam, 0.8)
    guided_support = guided >= np.quantile(guided, 0.8)
    union = np.logical_or(gradcam_support, guided_support).sum()
    intersection = np.logical_and(gradcam_support, guided_support).sum()
    overlap = float(intersection / union) if union else 0.0
    return combined, overlap


def make_hybrid_cam(
    model: Any,
    model_input: Any,
    original_rgb: np.ndarray,
    class_index: int,
    layer_name: str,
) -> dict[str, Any]:
    try:
        gradcam_heatmap = make_gradcam_heatmap(model, model_input, layer_name, class_index)
        guided_map = make_guided_backprop(
            model,
            model_input,
            original_rgb.shape[:2],
            class_index,
        )
        combined, overlap = fuse_heatmaps(gradcam_heatmap, guided_map)
        overlay = encode_overlay(original_rgb, combined, alpha=0.46)
    except Exception as exc:
        raise HybridCAMUnavailable(
            f"Could not generate the Guided Backpropagation/Grad-CAM fusion: {exc}"
        ) from exc
    return {
        "overlay": overlay,
        "guided_backprop": guided_map,
        "attribution_overlap": overlap,
        "fusion": (
            "pointwise product of independently max-normalized Grad-CAM "
            "and Guided Backpropagation maps"
        ),
    }
