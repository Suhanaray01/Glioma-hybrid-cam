from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SlicePrediction:
    index: int
    sequence: str
    used_for_classification: bool = True
    label: str = "glioma"
    glioma_probability: float = 0.5


@dataclass
class StudyResult:
    slices: list[SlicePrediction] = field(default_factory=list)
    aggregation: str = "max"
    final_label: str = "glioma"
    final_probability: float = 0.5
    warnings: list[str] = field(default_factory=list)

    def aggregate(self) -> "StudyResult":
        if not self.slices:
            return self
        probs = np.array([float(item.glioma_probability) for item in self.slices], dtype=np.float32)
        if self.aggregation == "mean":
            value = float(probs.mean())
        else:
            value = float(probs.max())
        self.final_probability = value
        self.final_label = "glioma" if value >= 0.5 else "normal"
        self.warnings = ["Only T2-FLAIR slices are used for classification; other sequences are context only."] if any(s.sequence != "T2FLAIR" for s in self.slices) else []
        return self


def aggregate_slice_predictions(slices: list[SlicePrediction], aggregation: str = "max") -> StudyResult:
    result = StudyResult(slices=slices, aggregation=aggregation)
    return result.aggregate()
