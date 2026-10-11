import numpy as np

from glioma_hc.preprocessing.pipeline import skull_removal_pipeline


def test_preprocessing_shape_and_range():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[40:220, 40:220] = 200

    output = skull_removal_pipeline(image)
    assert set(output) >= {"original", "opened", "brain", "clahe", "rgb"}
    assert output["brain"].shape == (256, 256)
    assert output["clahe"].shape == (256, 256)
    assert output["rgb"].shape == (256, 256, 3)
    assert output["brain"].min() >= 0
    assert output["brain"].max() <= 255
