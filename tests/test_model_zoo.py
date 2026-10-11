import pytest

pytest.importorskip("tensorflow")

from glioma_hc.training import model_zoo

build_model = model_zoo.build_model
list_supported_models = model_zoo.list_supported_models


def test_model_zoo_lists_supported_backbones():
    supported = list_supported_models()
    assert all(name in supported for name in model_zoo.REQUIRED_ENSEMBLE_MODELS)
    assert len(model_zoo.REQUIRED_ENSEMBLE_MODELS) == 8
    assert "inception_v3" in supported
    assert "densenet201" in supported
    assert "kaggle_cnn" in supported
    assert "attention_cnn" in supported
    assert "multiscale_cbam_cnn" in supported


@pytest.mark.parametrize("name", ["resnet18", "googlenet"])
def test_custom_ensemble_architectures_have_cam_ready_four_class_outputs(name):
    model = build_model(name, input_shape=(64, 64, 3), num_classes=4)

    assert model.output_shape == (None, 4)
    assert model.get_layer("target_conv_activation") is not None


def test_build_model_has_output_shape(monkeypatch):
    import tensorflow as tf

    def build_test_backbone(name, input_shape, include_top):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = tf.keras.layers.Conv2D(4, (1, 1))(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    monkeypatch.setattr(model_zoo, "build_backbone", build_test_backbone)
    model = build_model("inception_v3", input_shape=(224, 224, 3), num_classes=2)
    assert model.output_shape[-1] == 2


def test_attention_cnn_is_binary_and_cam_ready(tmp_path):
    import tensorflow as tf

    model = build_model("attention_cnn", input_shape=(32, 32, 3), num_classes=2)
    assert model.output_shape == (None, 2)
    assert model.get_layer("target_conv_activation") is not None
    assert model.get_layer("channel_attention") is not None
    assert model.get_layer("spatial_attention") is not None

    checkpoint = tmp_path / "attention.keras"
    model.save(checkpoint)
    loaded = tf.keras.models.load_model(checkpoint, compile=False)

    assert loaded.output_shape == (None, 2)
    assert loaded.get_layer("target_conv_activation") is not None


def test_kaggle_cnn_matches_four_class_notebook_architecture():
    model = build_model("kaggle_cnn", input_shape=(150, 150, 3), num_classes=4)

    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 4)
    assert model.get_layer("target_conv_activation") is not None


def test_multiscale_cbam_cnn_has_parallel_branches_cbam_and_gap(tmp_path):
    import tensorflow as tf

    model = build_model("multiscale_cbam_cnn", input_shape=(32, 32, 3), num_classes=2)
    layer_names = {layer.name for layer in model.layers}

    assert model.output_shape == (None, 2)
    assert "scale_block_2_depthwise_3x3" in layer_names
    assert "scale_block_2_depthwise_5x5" in layer_names
    assert "cbam_2_channel_attention" in layer_names
    assert "cbam_2_spatial_attention" in layer_names
    assert "gap" in layer_names
    assert "target_conv_activation" in layer_names

    checkpoint = tmp_path / "multiscale_cbam.keras"
    model.save(checkpoint)
    loaded = tf.keras.models.load_model(checkpoint, compile=False)
    assert loaded.output_shape == (None, 2)
