#!/usr/bin/env python
"""Convenience script for a quick end-to-end smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glioma_hc.dataset_builder.make_smoke_data import make_smoke_dataset
from glioma_hc.evaluation.evaluate import compute_metrics
from glioma_hc.training.train_all import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke run for GliomaScan-HC")
    parser.add_argument("--models", nargs="*", default=["inception_v3"])
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    make_smoke_dataset(project_root / "data" / "processed")
    training_summary = run_training(args.models, preset=args.preset, quick=True, project_root=project_root)

    true = [0, 1, 1, 0, 1, 1, 0, 1]
    probs = [0.12, 0.88, 0.91, 0.18, 0.73, 0.85, 0.11, 0.79]
    metrics = compute_metrics(true, probs)

    out_dir = project_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Quick run requested with models={args.models}, preset={args.preset}, quick={args.quick}")
    print(json.dumps({"training_summary": training_summary, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
