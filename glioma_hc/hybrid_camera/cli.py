from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from glioma_hc.hybrid_camera.ingest import load_image_to_8bit
from glioma_hc.hybrid_camera.quality import assess_capture_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-check an MRI capture or photograph before classification.")
    parser.add_argument("image", type=str, help="Path to an image or DICOM file")
    args = parser.parse_args()

    img = load_image_to_8bit(Path(args.image))
    result = assess_capture_quality(img)
    print(result)


if __name__ == "__main__":
    main()
