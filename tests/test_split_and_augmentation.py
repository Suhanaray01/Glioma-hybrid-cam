from glioma_hc.config import SETTINGS
from glioma_hc.dataset_builder.build_dataset import paper_split_indices, build_synthetic_paper_split
from glioma_hc.dataset_builder.import_kaggle_mri import import_dataset
from glioma_hc.dataset_builder.preprocessing import preprocess_mri_rgb


def test_paper_split_matches_expected_label_order():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    train_idx, test_idx = paper_split_indices(labels, seed=42)
    assert len(train_idx) + len(test_idx) == len(labels)
    assert SETTINGS.label_order == ["glioma", "meningioma", "notumor", "pituitary"]


def test_build_synthetic_paper_split_counts():
    summary = build_synthetic_paper_split()
    assert summary["train"] == 80
    assert summary["test"] == 20
    assert all(summary["counts"][label] == 20 for label in SETTINGS.label_order)


def test_kaggle_import_preserves_test_split_and_excludes_other_tumors(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"
    for split_name, count in (("Training", 5), ("Testing", 2)):
        for class_name in ("glioma", "notumor", "meningioma", "pituitary"):
            class_dir = source_dir / split_name / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                (class_dir / f"image_{index}.jpg").write_bytes(b"image")

    manifest = import_dataset(source_dir, output_dir, seed=17)

    assert manifest["counts"] == {
        "train": {
            "glioma": 4,
            "meningioma": 4,
            "notumor": 4,
            "pituitary": 4,
        },
        "val": {
            "glioma": 1,
            "meningioma": 1,
            "notumor": 1,
            "pituitary": 1,
        },
        "test": {
            "glioma": 2,
            "meningioma": 2,
            "notumor": 2,
            "pituitary": 2,
        },
    }
    assert manifest["label_order"] == list(SETTINGS.label_order)
    assert "does not specify MRI sequences" in manifest["modality_note"]
    assert (output_dir / "test" / "meningioma").is_dir()


def test_common_mri_preprocessing_returns_finite_rgb_uint8():
    import numpy as np

    pixels = np.zeros((20, 16, 3), dtype=np.uint8)
    pixels[5:15, 4:12] = 180

    processed = preprocess_mri_rgb(pixels)

    assert processed.shape == pixels.shape
    assert processed.dtype == np.uint8
    assert np.isfinite(processed).all()
    assert int(processed.max()) > int(processed.min())
