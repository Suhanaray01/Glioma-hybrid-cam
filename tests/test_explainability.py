import base64
import io

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf
from PIL import Image

from glioma_hc.explainability.gradcam import make_gradcam_overlay
from glioma_hc.explainability.guided_backprop import make_guided_backprop
from glioma_hc.explainability.hybrid_cam import fuse_heatmaps
from glioma_hc.inference import GRADCAM_LAYERS
from glioma_hc.training.model_zoo import build_model


def test_gradcam_returns_decodable_overlay_png():
    inputs = tf.keras.Input(shape=(8, 8, 3))
    features = tf.keras.layers.Conv2D(
        4,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="test_conv",
    )(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(pooled)
    model = tf.keras.Model(inputs, outputs)
    model_input = tf.random.uniform((1, 8, 8, 3), seed=42)
    original = np.full((8, 8, 3), 120, dtype=np.uint8)

    result = make_gradcam_overlay(model, model_input, original, "test_conv", 1)

    assert result.startswith("data:image/png;base64,")
    encoded = result.split(",", maxsplit=1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as overlay:
        assert overlay.size == (8, 8)


def test_resnet101_gradcam_target_layer_exists_on_keras_architecture():
    model = tf.keras.applications.ResNet101(
        include_top=False,
        weights=None,
        input_shape=(64, 64, 3),
    )

    target = model.get_layer(GRADCAM_LAYERS["resnet101"])

    assert target.output.shape.rank == 4


def test_hybrid_cam_fuses_maps_and_reports_overlap_separately():
    gradcam = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
    lime = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    fused, overlap = fuse_heatmaps(gradcam, lime)

    assert fused.shape == gradcam.shape
    assert float(fused.max()) == 1.0
    assert overlap == 1.0


def test_guided_backprop_returns_normalized_same_input_attribution():
    inputs = tf.keras.Input(shape=(8, 8, 3))
    features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(
        tf.keras.layers.GlobalAveragePooling2D()(features)
    )
    model = tf.keras.Model(inputs, outputs)
    model_input = tf.random.uniform((1, 8, 8, 3), seed=7)

    attribution = make_guided_backprop(model, model_input, (16, 12), class_index=1)

    assert attribution.shape == (16, 12)
    assert np.isfinite(attribution).all()
    assert 0 <= float(attribution.min()) <= float(attribution.max()) <= 1


@pytest.mark.parametrize("model_name", ["attention_cnn", "multiscale_cbam_cnn"])
def test_attention_cnn_target_layer_supports_gradcam(model_name):
    model = build_model(model_name, input_shape=(32, 32, 3), num_classes=2)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    model_input = tf.convert_to_tensor(image[None].astype(np.float32) / 255.0)

    overlay = make_gradcam_overlay(
        model,
        model_input,
        image,
        layer_name="target_conv_activation",
        class_index=1,
    )

    assert overlay.startswith("data:image/png;base64,")