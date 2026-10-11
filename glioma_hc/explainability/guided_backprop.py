from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - TensorFlow is optional for smoke scaffolding
    tf = None  # type: ignore[assignment]


class GuidedBackpropUnavailable(RuntimeError):
    pass


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="GliomaScanHC")
    def guided_relu(values: Any) -> Any:
        @tf.custom_gradient
        def guided(values_inner: Any) -> tuple[Any, Any]:
            activated = tf.nn.relu(values_inner)

            def gradient(upstream: Any) -> Any:
                gate_input = tf.cast(values_inner > 0, upstream.dtype)
                gate_upstream = tf.cast(upstream > 0, upstream.dtype)
                return upstream * gate_input * gate_upstream

            return activated, gradient

        return guided(values)

    @tf.keras.utils.register_keras_serializable(package="GliomaScanHC")
    class GuidedReLU(tf.keras.layers.Layer):
        def call(self, inputs: Any) -> Any:
            return guided_relu(inputs)


def encode_attribution(attribution: np.ndarray) -> str:
    normalized = np.asarray(attribution, dtype=np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.maximum(normalized, 0)
    maximum = float(normalized.max()) if normalized.size else 0.0
    if maximum > 0:
        normalized /= maximum
    colored = cv2.applyColorMap(
        np.uint8(np.clip(normalized, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    success, encoded = cv2.imencode(".png", colored)
    if not success:
        raise GuidedBackpropUnavailable("Could not encode the Guided Backpropagation map.")
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def make_guided_backprop(
    model: Any,
    model_input: Any,
    target_size: tuple[int, int],
    class_index: int,
) -> np.ndarray:
    if tf is None:
        raise GuidedBackpropUnavailable("TensorFlow is unavailable.")
    from glioma_hc.training.model_zoo import ChannelPool2D

    def clone_layer(layer: Any) -> Any:
        if isinstance(layer, tf.keras.layers.ReLU):
            config = layer.get_config()
            if config.get("negative_slope", 0) != 0 or config.get("max_value") is not None:
                raise GuidedBackpropUnavailable(
                    f"Guided backpropagation does not support configured ReLU layer {layer.name!r}."
                )
            return GuidedReLU(name=layer.name, trainable=layer.trainable)
        if isinstance(layer, tf.keras.layers.Activation) and layer.activation == tf.keras.activations.relu:
            config = layer.get_config()
            config["activation"] = guided_relu
            return layer.__class__.from_config(config)
        if getattr(layer, "activation", None) == tf.keras.activations.relu:
            config = layer.get_config()
            config["activation"] = guided_relu
            return layer.__class__.from_config(config)
        return layer.__class__.from_config(layer.get_config())

    try:
        guided_model = tf.keras.models.clone_model(
            model,
            clone_function=clone_layer,
        )
        guided_model.set_weights(model.get_weights())
        inputs = tf.convert_to_tensor(model_input)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = guided_model(inputs, training=False)
            score = predictions[:, class_index]
        gradients = tape.gradient(score, inputs)
        if gradients is None:
            raise GuidedBackpropUnavailable("Could not compute input gradients.")
        positive_gradient = tf.maximum(gradients[0], 0)
        attribution = tf.reduce_mean(positive_gradient, axis=-1).numpy()
    except GuidedBackpropUnavailable:
        raise
    except Exception as exc:
        raise GuidedBackpropUnavailable(
            f"Could not generate Guided Backpropagation: {exc}"
        ) from exc

    height, width = target_size
    attribution = cv2.resize(attribution, (width, height), interpolation=cv2.INTER_LINEAR)
    attribution = np.nan_to_num(attribution.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(attribution.max()) if attribution.size else 0.0
    if maximum > 0:
        attribution /= maximum
    return attribution
