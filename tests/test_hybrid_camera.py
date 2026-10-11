import numpy as np

from glioma_hc.hybrid_camera.camera_sim import camera_simulation
from glioma_hc.hybrid_camera.study import SlicePrediction, aggregate_slice_predictions


def test_camera_simulation_returns_uint8_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    sim = camera_simulation(image, severity=0.6)
    assert sim.dtype == np.uint8
    assert sim.shape == image.shape


def test_study_aggregation_handles_max_and_mean():
    slices = [
        SlicePrediction(index=0, sequence="T2FLAIR", label="glioma", glioma_probability=0.8),
        SlicePrediction(index=1, sequence="T2FLAIR", label="normal", glioma_probability=0.3),
    ]
    result = aggregate_slice_predictions(slices, aggregation="max")
    assert result.final_label == "glioma"
    assert result.final_probability >= 0.8

    mean_result = aggregate_slice_predictions(slices, aggregation="mean")
    assert mean_result.final_probability < 0.8
