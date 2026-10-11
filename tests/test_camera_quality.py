import numpy as np

from glioma_hc.hybrid_camera.quality import assess_capture_quality


def test_quality_gate_flags_blurry_and_overexposed_input():
    blurred = np.zeros((128, 128), dtype=np.uint8)
    result_blur = assess_capture_quality(blurred)
    assert result_blur["ok"] is False
    assert any("blur" in tip.lower() for tip in result_blur["tips"])

    overexposed = np.full((128, 128), 250, dtype=np.uint8)
    result_over = assess_capture_quality(overexposed)
    assert result_over["ok"] is False
    assert any("overexposed" in tip.lower() for tip in result_over["tips"])
