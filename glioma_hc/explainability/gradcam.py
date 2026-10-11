from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import cv2
import numpy as np


class GradCAMUnavailable(RuntimeError):
    pass


def make_gradcam_overlay(
    model: Any,
    model_input: Any,
    original_rgb: np.ndarray,
    layer_name: str,
    class_index: int,
) -> str:
    heatmap = make_gradcam_heatmap(model, model_input, layer_name, class_index)
    return encode_overlay(original_rgb, heatmap)


def make_gradcam_heatmap(
    model: Any,
    model_input: Any,
    layer_name: str,
    class_index: int,
) -> np.ndarray:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise GradCAMUnavailable("TensorFlow is unavailable.") from exc

    try:
        conv_layer = model.get_layer(layer_name)
    except (AttributeError, ValueError) as exc:
        raise GradCAMUnavailable(f"Grad-CAM layer {layer_name!r} was not found.") from exc

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_features, predictions = grad_model(model_input, training=False)
        target_score = predictions[:, class_index]
    gradients = tape.gradient(target_score, conv_features)
    if gradients is None:
        raise GradCAMUnavailable("Could not compute gradients for the selected convolutional layer.")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_features[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return np.asarray(heatmap, dtype=np.float32)


def encode_overlay(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> str:
    original = np.asarray(original_rgb, dtype=np.uint8)
    height, width = original.shape[:2]
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 1 - alpha, colored, alpha, 0)

    success, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise GradCAMUnavailable("Could not encode the Grad-CAM overlay.")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"