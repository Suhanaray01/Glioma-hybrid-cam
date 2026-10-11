import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from glioma_hc.inference import clear_model_cache
from glioma_hc.training.model_zoo import REQUIRED_ENSEMBLE_MODELS


client = TestClient(app)
LABEL_ORDER = ["glioma", "meningioma", "notumor", "pituitary"]


def _image_upload():
    image_bytes = io.BytesIO()
    Image.new("RGB", (12, 12)).save(image_bytes, format="PNG")
    return ("files", ("slice.png", image_bytes.getvalue(), "image/png"))


def _write_complete_ensemble_summary(models_dir, reports_dir):
    model_records = []
    for name in REQUIRED_ENSEMBLE_MODELS:
        model_path = models_dir / f"{name}.keras"
        model_path.touch()
        model_records.append({
            "name": name,
            "status": "trained",
            "model_path": str(model_path),
        })
    (reports_dir / "training_summary.json").write_text(json.dumps({
        "status": "ok",
        "quick": False,
        "label_order": LABEL_ORDER,
        "models": model_records,
        "ensemble": {
            "strategy": "soft_voting",
            "selection_split": "val",
            "model_order": list(REQUIRED_ENSEMBLE_MODELS),
            "model_weights": {name: 1 / len(REQUIRED_ENSEMBLE_MODELS) for name in REQUIRED_ENSEMBLE_MODELS},
        },
    }), encoding="utf-8")


def _write_kaggle_cnn_deployment(project_root, reports_dir, ensemble_models_dir):
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True)
    checkpoint = models_dir / "kaggle_cnn_paper.keras"
    checkpoint.touch()
    _write_complete_ensemble_summary(ensemble_models_dir, reports_dir)
    summary_path = reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["deployment"] = {
        "kind": "single_model",
        "model": "kaggle_cnn",
        "model_path": str(checkpoint.resolve()),
        "strategy": "single_model",
        "selection_split": "val",
        "preprocessing_profile": "kaggle_notebook_v1",
        "validation_metrics": {"accuracy": 0.94},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    checkpoint_stat = checkpoint.stat()
    report = {
        "model": checkpoint.name,
        "model_path": str(checkpoint.resolve()),
        "evaluation_split": "test",
        "evaluation_samples": 1600,
        "label_order": LABEL_ORDER,
        "metadata": {
            "preprocessing_profile": "kaggle_notebook_v1",
            "checkpoint_size_bytes": checkpoint_stat.st_size,
            "checkpoint_modified_time_ns": checkpoint_stat.st_mtime_ns,
        },
        "metrics": {"accuracy": 0.93125},
    }
    (reports_dir / "kaggle_cnn_paper_test.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return checkpoint


def test_health_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_loaded"] is False


def test_frontend_root_serves_application():
    response = client.get("/")

    assert response.status_code == 200
    assert "GliomaScan-HC" in response.text
    assert "showcase-metric" not in response.text

    script_response = client.get("/app.js")

    assert script_response.status_code == 200
    assert "['accuracy', 'Accuracy']" in script_response.text
    assert "targetValue.textContent = '98%'" not in script_response.text


def test_kaggle_cnn_deployment_serves_matching_metrics_and_predictions(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    reports_dir = project_root / "reports"
    ensemble_models_dir = project_root / "saved_models"
    reports_dir.mkdir(parents=True)
    ensemble_models_dir.mkdir(parents=True)
    checkpoint = _write_kaggle_cnn_deployment(
        project_root,
        reports_dir,
        ensemble_models_dir,
    )
    monkeypatch.setattr("backend.main.SETTINGS.project_root", project_root)
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", ensemble_models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    health = client.get("/api/health").json()
    assert health["predictor_ready"] is True
    assert health["ensemble_ready"] is False
    assert health["deployment"]["model"] == "kaggle_cnn"

    metrics = client.get("/api/metrics").json()
    assert metrics["report_current"] is True
    assert metrics["results"]["accuracy"] == pytest.approx(0.93125)

    observed_inputs = []

    class TestModel:
        input_shape = (None, 150, 150, 3)
        output_shape = (None, 4)

        def predict(self, batch, verbose=0):
            observed_inputs.append(np.asarray(batch))
            return np.tile([0.1, 0.2, 0.6, 0.1], (len(batch), 1))

    monkeypatch.setattr("glioma_hc.inference._load_keras_model", lambda path: TestModel())
    clear_model_cache()
    try:
        response = client.post(
            "/api/predict",
            files=[_image_upload()],
            data={"explain": "none"},
        )
    finally:
        clear_model_cache()

    assert response.status_code == 200
    prediction = response.json()
    assert prediction["label"] == "notumor"
    assert prediction["predictor"]["model"] == "kaggle_cnn"
    assert prediction["evaluation_metrics"]["accuracy"] == pytest.approx(0.93125)
    assert prediction["preprocessing"]["profile"] == "kaggle_notebook_v1"
    assert prediction["input"]["common_preprocessing_size"] is None
    assert observed_inputs[0].shape == (1, 150, 150, 3)
    assert np.min(observed_inputs[0]) >= 0
    assert np.max(observed_inputs[0]) <= 1


def test_predict_requires_trained_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", tmp_path / "models")
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", tmp_path / "reports")
    response = client.post("/api/predict", files=[_image_upload()])
    assert response.status_code == 503
    assert "eight-model training run" in response.json()["detail"]


def test_predict_rejects_upload_larger_than_configured_limit(monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.max_upload_bytes", 8)

    response = client.post("/api/predict", files=[_image_upload()])

    assert response.status_code == 413
    assert "configured limit" in response.json()["detail"]


def test_camera_quality_returns_validation_error_for_invalid_image():
    response = client.post(
        "/api/camera/quality",
        files={"file": ("invalid.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422
    assert "supported raster image" in response.json()["detail"]


def test_predict_rejects_unverified_sequence_claim():
    response = client.post(
        "/api/predict",
        files=[_image_upload()],
        data={"sequence": "T2FLAIR"},
    )

    assert response.status_code == 422
    assert "does not verify MRI sequence" in response.json()["detail"]


def test_predict_does_not_return_fixture_when_weights_exist(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", tmp_path)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", tmp_path / "reports")
    (tmp_path / "checkpoint.keras").touch()

    response = client.post("/api/predict", files=[_image_upload()])

    assert response.status_code == 503
    assert "eight-model training run" in response.json()["detail"]


def test_predict_returns_probabilities_from_recorded_model(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    _write_complete_ensemble_summary(models_dir, reports_dir)
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    class TestModel:
        name = "resnet18"
        input_shape = (None, 8, 8, 3)
        output_shape = (None, 4)

        def predict(self, batch, verbose=0):
            return np.tile([0.1, 0.2, 0.6, 0.1], (len(batch), 1))

    monkeypatch.setattr("glioma_hc.inference._load_keras_model", lambda path: TestModel())
    monkeypatch.setattr("glioma_hc.inference.preprocess_model_input", lambda batch, name: batch)
    image_bytes = io.BytesIO()
    Image.new("RGB", (12, 12)).save(image_bytes, format="PNG")

    response = client.post(
        "/api/predict",
        files=[("files", ("slice.png", image_bytes.getvalue(), "image/png"))],
        data={"explain": "none"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "notumor"
    assert result["confidence"] == pytest.approx(0.6)
    assert result["glioma_probability"] == pytest.approx(0.1)
    assert result["class_probabilities"] == {
        "glioma": pytest.approx(0.1),
        "meningioma": pytest.approx(0.2),
        "notumor": pytest.approx(0.6),
        "pituitary": pytest.approx(0.1),
    }
    assert result["models"][0]["probs"] == [0.1, 0.2, 0.6, 0.1]
    assert len(result["models"]) == 8
    assert result["ensemble"]["models_agreeing"] == 8
    assert result["input"]["original_width"] == 12
    assert result["original_image"].startswith("data:image/png;base64,")
    assert result["slices"][0]["sequence"] == "UNSPECIFIED"
    assert "MRI sequence was not specified by the dataset." in result["warnings"]
    assert result["explain"]["status"] == "disabled"


def test_predict_loads_only_validation_selected_single_model_and_caches_it(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    _write_complete_ensemble_summary(models_dir, reports_dir)
    summary_path = reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in summary["models"]:
        score = 0.9 if record["name"] == "resnet101" else 0.8
        record["validation_metrics"] = {
            "balanced_accuracy": score,
            "macro_f1": score,
            "macro_specificity": 0.95,
        }
    summary["ensemble"]["comparisons"] = {
        "soft_voting": {
            "metrics": {
                "balanced_accuracy": 0.85,
                "macro_f1": 0.84,
                "macro_specificity": 0.94,
            },
        },
    }
    summary["deployment"] = {
        "kind": "single_model",
        "model": "resnet101",
        "strategy": "single_model",
        "selection_split": "val",
        "validation_metrics": summary["models"][2]["validation_metrics"],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    class TestModel:
        input_shape = (None, 8, 8, 3)
        output_shape = (None, 4)

        def predict(self, batch, verbose=0):
            return np.tile([0.1, 0.2, 0.6, 0.1], (len(batch), 1))

    loads = []

    def load_model(path):
        loads.append(path)
        return TestModel()

    monkeypatch.setattr("glioma_hc.inference._load_keras_model", load_model)
    monkeypatch.setattr("glioma_hc.inference.preprocess_model_input", lambda batch, name: batch)
    clear_model_cache()
    try:
        responses = [
            client.post(
                "/api/predict",
                files=[_image_upload()],
                data={"explain": "none"},
            )
            for _ in range(2)
        ]
    finally:
        clear_model_cache()

    assert all(response.status_code == 200 for response in responses)
    result = responses[0].json()
    assert result["predictor"] == {
        "kind": "single_model",
        "model": "resnet101",
        "selection_split": "validation",
    }
    assert result["ensemble"]["strategy"] == "single_model"
    assert [model["name"] for model in result["models"]] == ["resnet101"]
    assert len(loads) == 1


def test_predict_hybrid_cam_returns_fused_overlay_and_overlap(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    _write_complete_ensemble_summary(models_dir, reports_dir)
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    class TestModel:
        name = "resnet18"
        input_shape = (None, 8, 8, 3)
        output_shape = (None, 4)

        def predict(self, batch, verbose=0):
            return np.tile([0.1, 0.2, 0.6, 0.1], (len(batch), 1))

    monkeypatch.setattr("glioma_hc.inference._load_keras_model", lambda path: TestModel())
    monkeypatch.setattr("glioma_hc.inference.preprocess_model_input", lambda batch, name: batch)
    monkeypatch.setattr("glioma_hc.inference.make_gradcam_overlay", lambda *args, **kwargs: "gradcam")
    monkeypatch.setattr(
        "glioma_hc.inference.make_hybrid_cam",
        lambda *args, **kwargs: {
            "overlay": "hybrid-cam",
            "guided_backprop": np.ones((12, 12), dtype=np.float32),
            "fusion": "pointwise product",
            "attribution_overlap": 0.4,
        },
    )

    response = client.post(
        "/api/predict",
        files=[_image_upload()],
        data={"explain": "hybrid_cam"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "notumor"
    assert result["glioma_probability"] == pytest.approx(0.1)
    assert result["explain"]["status"] == "generated"
    assert result["explain"]["hybrid_cam_overlay"] == "hybrid-cam"
    assert result["explain"]["guided_backprop"].startswith("data:image/png;base64,")
    assert result["explain"]["fusion_method"] == "pointwise product"
    assert result["explain"]["attribution_overlap"] == 0.4


def test_metrics_endpoint_reads_saved_test_results(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", tmp_path)
    (tmp_path / "test_results.json").write_text(json.dumps({
        "dataset": "test-dataset",
        "metrics": {"accuracy": 0.75},
    }), encoding="utf-8")

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["results"] == {}
    assert response.json()["evaluation"] == {}
    assert response.json()["report_current"] is False
    assert response.json()["stale_report_available"] is True


def test_metrics_endpoint_accepts_only_current_eight_model_test_report(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    _write_complete_ensemble_summary(models_dir, reports_dir)
    model_paths = {
        name: str((models_dir / f"{name}.keras").resolve())
        for name in REQUIRED_ENSEMBLE_MODELS
    }
    (reports_dir / "test_results.json").write_text(json.dumps({
        "model": "Eight-CNN ensemble",
        "models": list(REQUIRED_ENSEMBLE_MODELS),
        "model_checkpoint_paths": model_paths,
        "dataset": "test-dataset",
        "evaluation_split": "test",
        "ensemble": {
            "selection_split": "val",
            "strategy": "soft_voting",
        },
        "metrics": {"accuracy": 0.75},
    }), encoding="utf-8")
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["results"] == {"accuracy": 0.75}
    assert response.json()["report_current"] is True


def test_metrics_endpoint_serves_current_validation_selected_weighted_ensemble(tmp_path, monkeypatch):
    models_dir = tmp_path / "saved_models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    _write_complete_ensemble_summary(models_dir, reports_dir)
    model_paths = {
        name: f"saved_models/{name}.keras"
        for name in REQUIRED_ENSEMBLE_MODELS
    }
    summary_path = reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ensemble"]["strategy"] = "weighted_soft_voting"
    summary["project_root"] = str(tmp_path)
    for record in summary["models"]:
        record["model_path"] = model_paths[record["name"]]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (reports_dir / "weighted_soft_ensemble_test.json").write_text(json.dumps({
        "model": "Eight-CNN ensemble",
        "models": list(REQUIRED_ENSEMBLE_MODELS),
        "model_checkpoint_paths": model_paths,
        "evaluation_split": "test",
        "ensemble": {
            "selection_split": "val",
            "strategy": "weighted_soft_voting",
        },
        "deployment": {
            "kind": "ensemble",
            "model": None,
            "strategy": "weighted_soft_voting",
            "selection_split": "val",
        },
        "metrics": {"accuracy": 0.791875},
        "figures": {},
    }), encoding="utf-8")
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)
    monkeypatch.setattr("backend.main.SETTINGS.project_root", tmp_path)

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["report_current"] is True
    assert response.json()["results"]["accuracy"] == pytest.approx(0.791875)
