from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from glioma_hc.config import SETTINGS
from glioma_hc.dataset_builder.preprocessing import COMMON_IMAGE_SIZE, preprocess_mri_rgb

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - optional for smoke-only environments
    tf = None  # type: ignore[assignment]


def build_image_dataset(
    data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
    validation_split: float | None = None,
    subset: str | None = None,
    seed: int = 42,
    model_name: str | None = None,
    shuffle: bool = True,
    augment: bool = False,
    preprocessing_profile: str = "shared_mri",
):
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required for dataset loading and model training.")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Image dataset directory not found: {data_path}")
    if preprocessing_profile not in {"shared_mri", "kaggle_notebook_v1"}:
        raise ValueError(f"Unsupported preprocessing profile: {preprocessing_profile}")
    dataset_image_size = (
        image_size
        if preprocessing_profile == "kaggle_notebook_v1"
        else COMMON_IMAGE_SIZE
    )

    if validation_split is not None:
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(data_path),
            labels="inferred",
            class_names=SETTINGS.label_order,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=dataset_image_size,
            shuffle=shuffle,
            seed=seed,
            validation_split=validation_split,
            subset=subset,
        )
    else:
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(data_path),
            labels="inferred",
            class_names=SETTINGS.label_order,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=dataset_image_size,
            shuffle=shuffle,
            seed=seed,
        )

    def preprocess_and_resize(images: Any, labels: Any) -> tuple[Any, Any]:
        if preprocessing_profile == "kaggle_notebook_v1":
            images.set_shape((None, *image_size, 3))
            return tf.cast(images, tf.float32), labels
        processed = tf.numpy_function(_preprocess_batch, [images], tf.float32)
        processed.set_shape((None, *COMMON_IMAGE_SIZE, 3))
        return tf.image.resize(processed, image_size), labels

    dataset = dataset.map(preprocess_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(
                    factor=10 / 360,
                    fill_mode="nearest",
                    seed=seed,
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=0.002,
                    width_factor=0.002,
                    fill_mode="nearest",
                    seed=seed + 1,
                ),
                tf.keras.layers.RandomBrightness(
                    factor=0.15,
                    value_range=(0, 255),
                    seed=seed + 2,
                ),
            ]
        )
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if model_name is not None:
        dataset = dataset.map(
            lambda images, labels: (preprocess_model_input(images, model_name), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset


def _preprocess_batch(images: np.ndarray) -> np.ndarray:
    processed = np.stack([preprocess_mri_rgb(image) for image in images]).astype(np.float32)
    return processed


def preprocess_model_input(images: Any, model_name: str) -> Any:
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required to preprocess model inputs.")
    name = model_name.lower().replace("-", "_")
    if name in {"attention_cnn", "multiscale_cbam_cnn", "kaggle_cnn", "googlenet"}:
        return tf.cast(images, tf.float32) / 255.0
    if name == "efficientnet_b0":
        return tf.cast(images, tf.float32)
    application_name = "resnet" if name in {"resnet18", "resnet50", "resnet101"} else name
    applications = {
        "vgg16": tf.keras.applications.vgg16,
        "vgg19": tf.keras.applications.vgg19,
        "resnet": tf.keras.applications.resnet,
        "mobilenet_v2": tf.keras.applications.mobilenet_v2,
        "densenet": tf.keras.applications.densenet,
        "densenet201": tf.keras.applications.densenet,
        "efficientnet_b0": tf.keras.applications.efficientnet,
        "xception": tf.keras.applications.xception,
        "inception_v3": tf.keras.applications.inception_v3,
        "inception_resnet_v2": tf.keras.applications.inception_resnet_v2,
    }
    if name == "densenet121":
        application_name = "densenet"
    try:
        preprocess_input = applications[application_name].preprocess_input
    except KeyError as exc:
        raise ValueError(f"Unsupported model preprocessing: {model_name}") from exc
    return preprocess_input(tf.cast(images, tf.float32))
