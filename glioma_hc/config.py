from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = Path(os.getenv("GLIOMA_HC_MODELS_DIR", str(PROJECT_ROOT / "saved_models"))).expanduser()
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = reports_dir / "figures"
    seed: int = 42
    label_order: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
    image_size: dict[str, int] = {
        "default": 224,
        "xception": 299,
        "inception": 299,
        "kaggle_cnn": 150,
    }
    min_upload_bytes: int = 0
    max_upload_bytes: int = 25 * 1024 * 1024
    allowed_extensions: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".dicom", ".dcm", ".nii", ".nii.gz"}
    preferred_input_size: int = 150


SETTINGS = Settings()


def get_setting(name: str, default=None):
    value = os.getenv(name)
    if value is None:
        return getattr(SETTINGS, name, default)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.isdigit():
        return int(value)
    return value
