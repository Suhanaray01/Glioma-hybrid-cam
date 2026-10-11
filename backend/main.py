from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from glioma_hc.config import SETTINGS
from glioma_hc.evaluation.reporting import (
    is_current_ensemble_test_report,
    is_current_single_model_test_report,
)
from glioma_hc.hybrid_camera.quality import assess_capture_quality
from glioma_hc.inference import InferenceUnavailable, InvalidImage, predict_images
from glioma_hc.training.ensemble import deployment_from_training_summary
from glioma_hc.training.model_zoo import MODEL_SPECS, REQUIRED_ENSEMBLE_MODELS

app = FastAPI(title="GliomaScan-HC")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    model_dir = SETTINGS.models_dir
    model_available = model_dir.exists() and any(model_dir.glob("*.keras"))
    note = (
        "A model checkpoint is available; predictions require a completed compatible training run."
        if model_available
        else "Weights are required for real predictions."
    )
    return {"status": "ok", "models_loaded": model_available, "note": note}


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    model_records = {
        record.get("name"): record
        for record in summary.get("models", [])
        if isinstance(record, dict)
    }
    missing_models = [
        name
        for name in REQUIRED_ENSEMBLE_MODELS
        if name not in model_records
        or model_records[name].get("status") != "trained"
        or Path(model_records[name].get("model_path", "")).suffix != ".keras"
        or not Path(model_records[name].get("model_path", "")).is_file()
        or not Path(model_records[name].get("model_path", "")).resolve().is_relative_to(
            SETTINGS.models_dir.resolve()
        )
    ]
    ensemble = summary.get("ensemble", {})
    deployment = deployment_from_training_summary(summary)
    ensemble_ready = (
        summary.get("status") == "ok"
        and summary.get("quick") is False
        and summary.get("label_order") == SETTINGS.label_order
        and not missing_models
        and isinstance(deployment, dict)
        and deployment.get("kind") == "ensemble"
        and ensemble.get("selection_split") == "val"
        and ensemble.get("strategy")
        in {"hard_voting", "soft_voting", "weighted_soft_voting"}
        and ensemble.get("model_order") == list(REQUIRED_ENSEMBLE_MODELS)
    )
    selected_model_path = (
        Path(deployment.get("model_path", "")).resolve()
        if isinstance(deployment, dict) and deployment.get("kind") == "single_model"
        else None
    )
    predictor_ready = ensemble_ready or (
        summary.get("status") == "ok"
        and summary.get("quick") is False
        and summary.get("label_order") == SETTINGS.label_order
        and not missing_models
        and isinstance(deployment, dict)
        and deployment.get("kind") == "single_model"
        and deployment.get("selection_split") == "val"
        and deployment.get("model") in MODEL_SPECS
        and selected_model_path is not None
        and selected_model_path.suffix == ".keras"
        and (
            selected_model_path.is_relative_to(
                (SETTINGS.project_root / "models").resolve()
            )
            or selected_model_path.is_relative_to(SETTINGS.models_dir.resolve())
        )
        and selected_model_path.is_file()
    )
    return {
        "status": "ok",
        "models_loaded": len(REQUIRED_ENSEMBLE_MODELS) - len(missing_models),
        "models_required": len(REQUIRED_ENSEMBLE_MODELS),
        "missing_models": missing_models,
        "ensemble_ready": ensemble_ready,
        "predictor_ready": predictor_ready,
        "missing_weights": bool(missing_models),
        "training_summary": summary.get("status", "not_run"),
        "deployment": deployment,
    }


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"models": []}
    return {
        "models": summary.get("models", []),
        "saved_validation_metrics": [
            {
                "name": record.get("name"),
                "accuracy": record.get("final_val_accuracy"),
                "loss": record.get("val_loss"),
            }
            for record in summary.get("models", [])
        ],
        "ensemble_members": summary.get("ensemble", {}).get("model_order", []),
        "ensemble": summary.get("ensemble", {}),
        "deployment": deployment_from_training_summary(summary),
        "required_models": list(REQUIRED_ENSEMBLE_MODELS),
    }


@app.post("/api/camera/quality")
async def camera_quality(file: UploadFile = File(default=None)) -> dict[str, Any]:
    if file is None:
        return {"sharpness": 0.0, "brightness": 0.0, "glare_pct": 0.0, "roi_found": False, "ok": False, "tips": ["No file supplied."]}
    content = await file.read(SETTINGS.max_upload_bytes + 1)
    if len(content) > SETTINGS.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the configured limit ({SETTINGS.max_upload_bytes} bytes).",
        )
    if not content:
        return {"sharpness": 0.0, "brightness": 0.0, "glare_pct": 0.0, "roi_found": False, "ok": False, "tips": ["Empty upload."]}
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    import io

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Upload is not a supported raster image.") from exc
    arr = np.array(image)
    return assess_capture_quality(arr)


@app.post("/api/predict")
async def predict(
    files: list[UploadFile] = File(default_factory=list),
    sequence: str = Form("UNSPECIFIED"),
    source: str = Form("upload"),
    explain: str = Form("gradcam"),
    aggregation: str = Form("max"),
) -> dict[str, Any]:
    if sequence.upper() != "UNSPECIFIED":
        raise HTTPException(
            status_code=422,
            detail="The trained dataset does not verify MRI sequence; submit with sequence UNSPECIFIED.",
        )
    image_files = []
    total_bytes = 0
    for file in files:
        remaining = SETTINGS.max_upload_bytes - total_bytes
        content = await file.read(remaining + 1)
        if len(content) > remaining:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Combined uploaded images exceed the configured limit "
                    f"({SETTINGS.max_upload_bytes} bytes)."
                ),
            )
        total_bytes += len(content)
        image_files.append((file.filename or "uploaded image", content))
    try:
        return await run_in_threadpool(predict_images, image_files, aggregation, explain, sequence)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics")
def api_metrics() -> dict[str, Any]:
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    deployment = deployment_from_training_summary(summary)
    if isinstance(deployment, dict) and deployment.get("kind") == "single_model":
        results_path = SETTINGS.reports_dir / "kaggle_cnn_paper_test.json"
    else:
        results_path = SETTINGS.reports_dir / "weighted_soft_ensemble_test.json"
        if not results_path.is_file():
            results_path = SETTINGS.reports_dir / "final_test_results.json"
        if not results_path.is_file():
            results_path = SETTINGS.reports_dir / "test_results.json"
    paper_path = SETTINGS.reports_dir / "paper_reference.json"
    report = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    current_report = (
        is_current_single_model_test_report(report, deployment)
        if isinstance(deployment, dict) and deployment.get("kind") == "single_model"
        else is_current_ensemble_test_report(report, summary)
    )
    evaluation = report if current_report else {}
    paper_reference = json.loads(paper_path.read_text(encoding="utf-8")) if paper_path.exists() else {}
    figure_urls = {}
    if current_report:
        for name, relative_path in evaluation.get("figures", {}).items():
            candidate = (results_path.parent / relative_path).resolve()
            if (
                candidate.parent == (results_path.parent / "figures").resolve()
                and candidate.is_file()
                and candidate.suffix.lower() == ".png"
            ):
                figure_urls[name] = f"/api/figures/{candidate.name}"
    return {
        "results": evaluation.get("metrics", {}),
        "evaluation": evaluation,
        "report_current": current_report,
        "stale_report_available": bool(report),
        "paper_reference": paper_reference,
        "prediction_confidence": evaluation.get("prediction_confidence", {}),
        "figures": figure_urls,
    }


@app.get("/api/figures/{filename}")
def api_figure(filename: str) -> FileResponse:
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Evaluation figure not found.")
    allowed_directories = (
        (SETTINGS.reports_dir / "figures").resolve(),
        SETTINGS.figures_dir.resolve(),
    )
    figure_path = next(
        (
            directory / filename
            for directory in allowed_directories
            if (directory / filename).is_file()
        ),
        None,
    )
    if figure_path is None or figure_path.resolve().parent not in allowed_directories:
        raise HTTPException(status_code=404, detail="Evaluation figure not found.")
    return FileResponse(figure_path, media_type="image/png")


@app.post("/api/report")
def api_report() -> dict[str, Any]:
    return {"status": "not_implemented", "message": "PDF report generation can be added when report data exists."}


app.mount("/", StaticFiles(directory=SETTINGS.project_root / "frontend", html=True), name="frontend")
