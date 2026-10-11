from pathlib import Path

import pytest

from glioma_hc.config import SETTINGS
from glioma_hc.training import train_all


def test_quick_training_run_creates_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)
    result = train_all.run_training(
        ["inception_v3"],
        preset="fast",
        quick=True,
        project_root=tmp_path,
    )

    assert result["status"] == "skipped"
    assert (tmp_path / "reports" / "training_summary.json").exists()
    assert "models" in result
    assert result["label_order"] == SETTINGS.label_order
    assert result["training_strategy"] == "frozen_backbone_feature_extraction"
    assert result["training_data_dir"] == str(tmp_path / "data" / "smoke" / "processed")
    assert not (tmp_path / "data" / "processed").exists()


def test_default_training_uses_kaggle_cnn(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)
    result = train_all.run_training(quick=True, project_root=tmp_path)

    assert result["training_strategy"] == "eight_cnn_transfer_ensemble"


def test_image_dataset_uses_configured_label_order(tmp_path):
    pytest.importorskip("tensorflow")
    from PIL import Image
    import numpy as np

    from glioma_hc.config import SETTINGS
    from glioma_hc.training.data_loader import build_image_dataset

    pixel_values = {"glioma": 255, "meningioma": 170, "notumor": 0, "pituitary": 85}
    for label, pixel_value in pixel_values.items():
        class_dir = tmp_path / label
        class_dir.mkdir()
        Image.new("RGB", (8, 8), color=(pixel_value,) * 3).save(class_dir / "sample.png")

    dataset = build_image_dataset(
        tmp_path,
        image_size=(8, 8),
        batch_size=4,
        model_name="inception_v3",
    )

    images, labels = next(iter(dataset))
    pixel_means = images.numpy().mean(axis=(1, 2, 3))
    label_values = labels.numpy()
    assert label_values[np.argmin(pixel_means)] == SETTINGS.label_order.index("notumor")
    assert label_values[np.argmax(pixel_means)] == SETTINGS.label_order.index("glioma")
    assert float(pixel_means.min()) >= -1.0
    assert float(pixel_means.min()) < -0.9
    assert float(pixel_means.max()) == 1.0


def test_attention_cnn_input_is_scaled_to_unit_range():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from glioma_hc.training.data_loader import preprocess_model_input

    images = tf.constant([[[[0.0, 127.5, 255.0]]]])
    processed = preprocess_model_input(images, "attention_cnn")

    assert processed.numpy().tolist() == [[[[0.0, 0.5, 1.0]]]]


def test_multiscale_cbam_input_is_scaled_to_unit_range():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from glioma_hc.training.data_loader import preprocess_model_input

    images = tf.constant([[[[0.0, 127.5, 255.0]]]])
    processed = preprocess_model_input(images, "multiscale_cbam_cnn")

    assert processed.numpy().tolist() == [[[[0.0, 0.5, 1.0]]]]


def test_kaggle_cnn_input_is_scaled_to_unit_range():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from glioma_hc.training.data_loader import preprocess_model_input

    images = tf.constant([[[[0.0, 127.5, 255.0]]]])
    processed = preprocess_model_input(images, "kaggle_cnn")

    assert processed.numpy().tolist() == [[[[0.0, 0.5, 1.0]]]]


def test_kaggle_training_augmentation_preserves_images_and_labels(tmp_path):
    pytest.importorskip("tensorflow")
    import numpy as np
    from PIL import Image

    from glioma_hc.training.data_loader import build_image_dataset

    for index, label in enumerate(SETTINGS.label_order):
        class_dir = tmp_path / label
        class_dir.mkdir()
        Image.new("RGB", (12, 12), color=(60 * index, 60 * index, 60 * index)).save(
            class_dir / "sample.png"
        )

    dataset = build_image_dataset(
        tmp_path,
        image_size=(12, 12),
        batch_size=4,
        seed=7,
        model_name="kaggle_cnn",
        augment=True,
        shuffle=False,
    )

    images, labels = next(iter(dataset))
    assert images.shape == (4, 12, 12, 3)
    assert sorted(labels.numpy().tolist()) == list(range(len(SETTINGS.label_order)))
    assert np.isfinite(images.numpy()).all()
    assert float(images.numpy().min()) >= 0
    assert float(images.numpy().max()) <= 1


def test_real_training_skips_without_raw_source_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)
    for label in SETTINGS.label_order:
        class_dir = tmp_path / "data" / "processed" / "train" / label
        class_dir.mkdir(parents=True)
        (class_dir / f"{label}_train_0.png").touch()

    result = train_all.run_training(
        ["inception_v3"],
        preset="paper",
        quick=False,
        project_root=tmp_path,
    )

    assert result["status"] == "skipped"
    assert "data/raw" in result["note"]
    assert result["models"] == []


def test_training_summary_reports_per_class_counts_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)
    for split, images_per_class in (("train", 3), ("val", 1), ("test", 2)):
        for label in SETTINGS.label_order:
            class_dir = tmp_path / "processed" / split / label
            class_dir.mkdir(parents=True)
            for index in range(images_per_class):
                (class_dir / f"{label}_{index}.png").touch()

    result = train_all.run_training(
        ["resnet18"],
        project_root=tmp_path,
        data_dir_override=tmp_path / "processed",
    )

    assert result["dataset_counts"]["train"]["glioma"] == 3
    assert result["dataset_counts"]["val"]["pituitary"] == 1
    assert result["dataset_counts"]["test"]["meningioma"] == 2


def test_attention_model_summary_does_not_claim_frozen_backbone(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)

    result = train_all.run_training(
        ["attention_cnn"],
        preset="attention",
        quick=True,
        project_root=tmp_path,
    )

    assert result["training_strategy"] == "attention_cnn_from_scratch"
