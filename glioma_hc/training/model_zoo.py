from __future__ import annotations

from typing import Any, TYPE_CHECKING

try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:  # pragma: no cover - runtime dependency is optional for smoke scaffolding
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]


MODEL_SPECS = {
    "kaggle_cnn": {"input_size": 150, "backend": "kaggle_cnn"},
    "resnet18": {"input_size": 224, "backend": "keras_resnet18", "pretrained": False},
    "resnet50": {"input_size": 224, "backend": "resnet50", "pretrained": True},
    "resnet101": {"input_size": 224, "backend": "resnet101", "pretrained": True},
    "googlenet": {"input_size": 224, "backend": "keras_googlenet", "pretrained": False},
    "efficientnet_b0": {"input_size": 224, "backend": "efficientnet_b0", "pretrained": True},
    "mobilenet_v2": {"input_size": 224, "backend": "mobilenet_v2", "pretrained": True},
    "densenet121": {"input_size": 224, "backend": "densenet121", "pretrained": True},
    "inception_v3": {"input_size": 299, "backend": "inception_v3", "pretrained": True},
    "vgg16": {"input_size": 224, "backend": "vgg16", "pretrained": True},
    "vgg19": {"input_size": 224, "backend": "vgg19", "pretrained": True},
    "densenet201": {"input_size": 224, "backend": "densenet201", "pretrained": True},
    "xception": {"input_size": 299, "backend": "xception", "pretrained": True},
    "inception_resnet_v2": {"input_size": 299, "backend": "inception_resnet_v2", "pretrained": True},
    "attention_cnn": {"input_size": 224, "backend": "keras_attention_cnn"},
    "multiscale_cbam_cnn": {"input_size": 224, "backend": "keras_multiscale_cbam_cnn"},
}
REQUIRED_ENSEMBLE_MODELS = (
    "resnet18",
    "resnet50",
    "resnet101",
    "googlenet",
    "efficientnet_b0",
    "mobilenet_v2",
    "inception_v3",
    "densenet121",
)


def _require_tensorflow() -> None:
    if tf is None or keras is None:
        raise ModuleNotFoundError(
            "TensorFlow is required to build image models. Install it in a machine with sufficient space or use a dedicated training environment."
        )


if keras is not None:
    @keras.utils.register_keras_serializable(package="GliomaScanHC")
    class ChannelPool2D(keras.layers.Layer):
        def __init__(self, mode: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            if mode not in {"max", "mean"}:
                raise ValueError("mode must be 'max' or 'mean'")
            self.mode = mode

        def call(self, inputs: Any) -> Any:
            if self.mode == "max":
                return tf.reduce_max(inputs, axis=-1, keepdims=True)
            return tf.reduce_mean(inputs, axis=-1, keepdims=True)

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "mode": self.mode}
else:  # pragma: no cover - only used when TensorFlow is not installed
    ChannelPool2D = None  # type: ignore[assignment,misc]


def build_backbone(name: str, input_shape: tuple[int, int, int] = (224, 224, 3), include_top: bool = False) -> Any:
    _require_tensorflow()
    name = name.lower().replace("-", "_")
    if name == "vgg16":
        return keras.applications.VGG16(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "vgg19":
        return keras.applications.VGG19(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "resnet50":
        return keras.applications.ResNet50(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "resnet101":
        return keras.applications.ResNet101(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "efficientnet_b0":
        return keras.applications.EfficientNetB0(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "mobilenet_v2":
        return keras.applications.MobileNetV2(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "densenet121":
        return keras.applications.DenseNet121(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "densenet201":
        return keras.applications.DenseNet201(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "xception":
        return keras.applications.Xception(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "inception_v3":
        return keras.applications.InceptionV3(include_top=include_top, weights="imagenet", input_shape=input_shape)
    if name == "inception_resnet_v2":
        return keras.applications.InceptionResNetV2(include_top=include_top, weights="imagenet", input_shape=input_shape)
    raise ValueError(f"Unsupported model: {name}")


def build_model(
    name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    freeze_backbone: bool = False,
) -> Any:
    _require_tensorflow()
    backbone_name = name.lower().replace("-", "_")
    if backbone_name == "kaggle_cnn":
        return _build_kaggle_cnn(input_shape, num_classes)
    if backbone_name == "resnet18":
        return _build_resnet18(input_shape, num_classes)
    if backbone_name == "googlenet":
        return _build_googlenet(input_shape, num_classes)
    if backbone_name == "attention_cnn":
        return _build_attention_cnn(input_shape, num_classes)
    if backbone_name == "multiscale_cbam_cnn":
        return _build_multiscale_cbam_cnn(input_shape, num_classes)

    base = build_backbone(backbone_name, input_shape=input_shape, include_top=False)
    base.trainable = not freeze_backbone
    x = base.output
    if backbone_name in {"vgg16", "vgg19"}:
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=base.input, outputs=x, name=backbone_name)
    return model


def _residual_block(inputs: Any, filters: int, stride: int, name: str) -> Any:
    residual = inputs
    x = keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1"
    )(inputs)
    x = keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or inputs.shape[-1] != filters:
        residual = keras.layers.Conv2D(
            filters, 1, strides=stride, use_bias=False, name=f"{name}_shortcut_conv"
        )(residual)
        residual = keras.layers.BatchNormalization(name=f"{name}_shortcut_bn")(residual)
    x = keras.layers.Add(name=f"{name}_add")([x, residual])
    return keras.layers.ReLU(name=f"{name}_out")(x)


def _build_resnet18(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    inputs = keras.Input(shape=input_shape, name="image")
    x = keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
    for stage, filters in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            stride = 2 if stage > 1 and block == 0 else 1
            x = _residual_block(x, filters, stride, f"stage{stage}_block{block + 1}")
    x = keras.layers.Activation("linear", name="target_conv_activation")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="resnet18")


def _inception_block(
    inputs: Any,
    filters: tuple[int, int, int, int, int, int],
    name: str,
) -> Any:
    one = keras.layers.Conv2D(filters[0], 1, padding="same", activation="relu", name=f"{name}_1x1")(inputs)

    three = keras.layers.Conv2D(filters[1], 1, padding="same", activation="relu", name=f"{name}_3x3_reduce")(inputs)
    three = keras.layers.Conv2D(filters[2], 3, padding="same", activation="relu", name=f"{name}_3x3")(three)

    five = keras.layers.Conv2D(filters[3], 1, padding="same", activation="relu", name=f"{name}_5x5_reduce")(inputs)
    five = keras.layers.Conv2D(filters[4], 5, padding="same", activation="relu", name=f"{name}_5x5")(five)

    pool = keras.layers.MaxPooling2D(3, strides=1, padding="same", name=f"{name}_pool")(inputs)
    pool = keras.layers.Conv2D(filters[5], 1, padding="same", activation="relu", name=f"{name}_pool_proj")(pool)
    return keras.layers.Concatenate(name=f"{name}_concat")([one, three, five, pool])


def _build_googlenet(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    inputs = keras.Input(shape=input_shape, name="image")
    x = keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu", name="stem_conv")(inputs)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
    x = keras.layers.Conv2D(64, 1, activation="relu", name="pre_inception_reduce")(x)
    x = keras.layers.Conv2D(192, 3, padding="same", activation="relu", name="pre_inception_conv")(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same", name="pre_inception_pool")(x)
    x = _inception_block(x, (64, 96, 128, 16, 32, 32), "inception_3a")
    x = _inception_block(x, (128, 128, 192, 32, 96, 64), "inception_3b")
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same", name="inception_pool_3")(x)
    x = _inception_block(x, (192, 96, 208, 16, 48, 64), "inception_4a")
    x = _inception_block(x, (160, 112, 224, 24, 64, 64), "inception_4b")
    x = _inception_block(x, (128, 128, 256, 24, 64, 64), "inception_4c")
    x = _inception_block(x, (112, 144, 288, 32, 64, 64), "inception_4d")
    x = _inception_block(x, (256, 160, 320, 32, 128, 128), "inception_4e")
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same", name="inception_pool_4")(x)
    x = _inception_block(x, (256, 160, 320, 32, 128, 128), "inception_5a")
    x = _inception_block(x, (384, 192, 384, 48, 128, 128), "inception_5b")
    x = keras.layers.Activation("linear", name="target_conv_activation")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="googlenet")


def _build_kaggle_cnn(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for index, filters in enumerate((32, 64, 128, 128), start=1):
        layer_name = "target_conv_activation" if index == 4 else f"conv{index}"
        x = keras.layers.Conv2D(
            filters,
            kernel_size=(4, 4),
            activation="relu",
            name=layer_name,
        )(x)
        if index < 4:
            x = keras.layers.MaxPooling2D(pool_size=(3, 3), name=f"pool{index}")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5, seed=42)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="kaggle_cnn")


def _build_attention_cnn(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    inputs = keras.Input(shape=input_shape, name="image")
    x = keras.layers.Conv2D(32, 3, padding="same", name="conv1")(inputs)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(128, 3, padding="same", name="target_conv")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.ReLU(name="target_conv_activation")(x)

    channel = keras.layers.GlobalAveragePooling2D()(x)
    channel = keras.layers.Reshape((1, 1, 128))(channel)
    channel = keras.layers.Conv2D(32, 1, activation="relu", name="channel_reduce")(channel)
    channel = keras.layers.Conv2D(128, 1, activation="sigmoid", name="channel_attention")(channel)
    x = keras.layers.Multiply()([x, channel])

    spatial_max = ChannelPool2D("max", name="spatial_max")(x)
    spatial_mean = ChannelPool2D("mean", name="spatial_mean")(x)
    spatial = keras.layers.Concatenate(axis=-1)([spatial_max, spatial_mean])
    spatial = keras.layers.Conv2D(
        1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        name="spatial_attention",
    )(spatial)
    x = keras.layers.Multiply()([x, spatial])
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="attention_cnn")


def _cbam_block(inputs: Any, channels: int, name: str) -> Any:
    hidden_channels = max(channels // 8, 8)
    shared_reduce = keras.layers.Dense(hidden_channels, activation="relu", name=f"{name}_channel_reduce")
    shared_expand = keras.layers.Dense(channels, name=f"{name}_channel_expand")
    average = keras.layers.GlobalAveragePooling2D(name=f"{name}_channel_avg_pool")(inputs)
    maximum = keras.layers.GlobalMaxPooling2D(name=f"{name}_channel_max_pool")(inputs)
    average = shared_expand(shared_reduce(average))
    maximum = shared_expand(shared_reduce(maximum))
    channel = keras.layers.Add(name=f"{name}_channel_add")([average, maximum])
    channel = keras.layers.Activation("sigmoid", name=f"{name}_channel_attention")(channel)
    channel = keras.layers.Reshape((1, 1, channels), name=f"{name}_channel_reshape")(channel)
    attended = keras.layers.Multiply(name=f"{name}_channel_apply")([inputs, channel])

    spatial_max = ChannelPool2D("max", name=f"{name}_spatial_max")(attended)
    spatial_mean = ChannelPool2D("mean", name=f"{name}_spatial_mean")(attended)
    spatial = keras.layers.Concatenate(axis=-1, name=f"{name}_spatial_pool_concat")([spatial_max, spatial_mean])
    spatial = keras.layers.Conv2D(
        1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        name=f"{name}_spatial_attention",
    )(spatial)
    return keras.layers.Multiply(name=f"{name}_spatial_apply")([attended, spatial])


def _multiscale_depthwise_block(inputs: Any, channels: int, name: str) -> Any:
    branch_3 = keras.layers.SeparableConv2D(
        channels,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise_3x3",
    )(inputs)
    branch_5 = keras.layers.SeparableConv2D(
        channels,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise_5x5",
    )(inputs)
    merged = keras.layers.Concatenate(name=f"{name}_multiscale_concat")([branch_3, branch_5])
    merged = keras.layers.Conv2D(channels, 1, use_bias=False, name=f"{name}_pointwise_fusion")(merged)
    merged = keras.layers.BatchNormalization(name=f"{name}_batch_norm")(merged)
    return keras.layers.ReLU(name=f"{name}_activation")(merged)


def _build_multiscale_cbam_cnn(input_shape: tuple[int, int, int], num_classes: int) -> Any:
    inputs = keras.Input(shape=input_shape, name="image")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_batch_norm")(x)
    x = keras.layers.ReLU(name="stem_activation")(x)
    x = keras.layers.MaxPooling2D(2, name="stem_pool")(x)

    x = _multiscale_depthwise_block(x, 64, "scale_block_1")
    x = _cbam_block(x, 64, "cbam_1")
    x = keras.layers.MaxPooling2D(2, name="scale_pool_1")(x)

    x = _multiscale_depthwise_block(x, 128, "scale_block_2")
    x = keras.layers.Activation("linear", name="target_conv_activation")(x)
    x = _cbam_block(x, 128, "cbam_2")
    x = keras.layers.MaxPooling2D(2, name="scale_pool_2")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="multiscale_cbam_cnn")


def list_supported_models() -> list[str]:
    return sorted(MODEL_SPECS)
