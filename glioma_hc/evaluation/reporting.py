from __future__ import annotations

from pathlib import Path
from typing import Any

from glioma_hc.config import SETTINGS
from glioma_hc.training.ensemble import deployment_from_training_summary
from glioma_hc.training.model_zoo import MODEL_SPECS, REQUIRED_ENSEMBLE_MODELS


def _resolve_checkpoint_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = SETTINGS.project_root / path
    return str(path.resolve())


def is_current_ensemble_test_report(
    report: dict[str, Any],
    training_summary: dict[str, Any],
) -> bool:
    ensemble = training_summary.get("ensemble", {})
    report_ensemble = report.get("ensemble", {})
    records = {
        record.get("name"): record
        for record in training_summary.get("models", [])
        if isinstance(record, dict)
    }
    expected_paths = {
        name: _resolve_checkpoint_path(records[name].get("model_path", ""))
        for name in ensemble.get("model_order", [])
        if name in records
    }
    report_paths = report.get("model_checkpoint_paths", {})
    if not isinstance(report_paths, dict) or not all(
        isinstance(name, str) and isinstance(path, str)
        for name, path in report_paths.items()
    ):
        return False
    resolved_report_paths = {
        name: _resolve_checkpoint_path(path)
        for name, path in report_paths.items()
    }
    deployment = deployment_from_training_summary(training_summary)
    has_validation_metrics = any(
        isinstance(records.get(name, {}).get("validation_metrics"), dict)
        and "balanced_accuracy" in records[name]["validation_metrics"]
        for name in REQUIRED_ENSEMBLE_MODELS
    )
    if deployment is None:
        return False
    report_deployment = report.get("deployment", {})
    if not report_deployment and not has_validation_metrics:
        report_deployment = deployment
    return (
        training_summary.get("status") == "ok"
        and training_summary.get("quick") is False
        and len(expected_paths) == len(REQUIRED_ENSEMBLE_MODELS)
        and report.get("evaluation_split") == "test"
        and report.get("models") == list(REQUIRED_ENSEMBLE_MODELS)
        and resolved_report_paths == expected_paths
        and report_ensemble.get("selection_split") == "val"
        and report_ensemble.get("strategy") == ensemble.get("strategy")
        and report_deployment.get("selection_split") == "val"
        and report_deployment.get("kind") == deployment.get("kind")
        and report_deployment.get("model") == deployment.get("model")
        and report_deployment.get("strategy") == deployment.get("strategy")
        and report.get("model")
        == (deployment.get("model") or "Eight-CNN ensemble")
    )


def is_current_single_model_test_report(
    report: dict[str, Any],
    deployment: dict[str, Any] | None,
) -> bool:
    if not isinstance(deployment, dict) or deployment.get("kind") != "single_model":
        return False
    model_name = deployment.get("model")
    model_path_value = deployment.get("model_path")
    if (
        model_name not in MODEL_SPECS
        or not isinstance(model_path_value, str)
        or deployment.get("selection_split") != "val"
    ):
        return False

    model_path = Path(model_path_value).resolve()
    allowed_root = (SETTINGS.project_root / "models").resolve()
    if (
        model_path.suffix != ".keras"
        or not model_path.is_relative_to(allowed_root)
        or not model_path.is_file()
    ):
        return False
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    return (
        report.get("evaluation_split") == "test"
        and report.get("label_order") == SETTINGS.label_order
        and report.get("model_path") == str(model_path)
        and report.get("model") == model_path.name
        and metadata.get("preprocessing_profile")
        == deployment.get("preprocessing_profile")
        and metadata.get("checkpoint_size_bytes") == model_path.stat().st_size
        and metadata.get("checkpoint_modified_time_ns") == model_path.stat().st_mtime_ns
    )
