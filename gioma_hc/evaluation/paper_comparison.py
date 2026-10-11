from __future__ import annotations

import json
from pathlib import Path


def compare_metrics(our_results: dict, paper_path: str | Path) -> dict:
    paper = json.loads(Path(paper_path).read_text(encoding="utf-8"))
    deltas: dict[str, dict[str, float | str | None]] = {}
    for model_name, metrics in our_results.items():
        paper_model = paper["paper"].get(model_name, {})
        delta = {}
        for key in {"acc", "sens", "spec", "auc"}:
            if key in metrics and key in paper_model:
                delta[key] = float(metrics[key]) - float(paper_model[key])
        deltas[model_name] = delta
    return deltas
