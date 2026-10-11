from __future__ import annotations

import numpy as np

from glioma_hc.config import SETTINGS
from glioma_hc.training.model_zoo import REQUIRED_ENSEMBLE_MODELS


def _validate_probability_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(prob_matrix, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(SETTINGS.label_order):
        raise ValueError("Probability matrix must have one column per configured class.")
    if probabilities.shape[0] == 0:
        raise ValueError("At least one model prediction is required.")
    _validate_probability_values(probabilities)
    return probabilities


def _validate_probability_values(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Model probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-3):
        raise ValueError("Each model prediction must sum to one.")


def hard_vote(prob_matrix: np.ndarray) -> tuple[str, int, list[int]]:
    probabilities = _validate_probability_matrix(prob_matrix)
    predictions = np.argmax(probabilities, axis=1)
    vote_counts = np.bincount(predictions, minlength=len(SETTINGS.label_order))
    winner = int(np.argmax(vote_counts))
    return SETTINGS.label_order[winner], int(np.max(vote_counts)), vote_counts.tolist()


def soft_vote(prob_matrix: np.ndarray) -> tuple[str, np.ndarray]:
    probabilities = _validate_probability_matrix(prob_matrix)
    summed = np.sum(probabilities, axis=0)
    winner = int(np.argmax(summed))
    return SETTINGS.label_order[winner], summed


def majority_vote(probabilities: list[np.ndarray], threshold: float = 0.5) -> dict:
    arr = np.asarray(probabilities)
    hard_label, votes, counts = hard_vote(arr)
    soft_label, soft_probs = soft_vote(arr)
    return {"hard": {"label": hard_label, "votes": votes, "counts": counts}, "soft": {"label": soft_label, "probs": soft_probs.tolist(), "threshold": threshold}}


def combine_model_probabilities(
    model_probabilities: np.ndarray,
    strategy: str,
    model_weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    probabilities = np.asarray(model_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] == 0:
        raise ValueError("Model probabilities must have shape (models, samples, classes).")
    if probabilities.shape[2] != len(SETTINGS.label_order):
        raise ValueError("Probability matrix must have one column per configured class.")
    _validate_probability_values(probabilities)

    if strategy == "hard_voting":
        predictions = np.argmax(probabilities, axis=2)
        votes = np.eye(len(SETTINGS.label_order), dtype=np.float64)[predictions]
        combined = votes.mean(axis=0)
    elif strategy == "soft_voting":
        combined = probabilities.mean(axis=0)
    elif strategy == "weighted_soft_voting":
        weights = np.asarray(model_weights, dtype=np.float64)
        if weights.shape != (probabilities.shape[0],):
            raise ValueError("Weighted soft voting requires one weight per model.")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Model weights must be finite, non-negative, and have a positive sum.")
        combined = np.average(probabilities, axis=0, weights=weights)
    else:
        raise ValueError("Unknown ensemble strategy.")

    row_sums = combined.sum(axis=1, keepdims=True)
    return np.divide(combined, row_sums, out=np.zeros_like(combined), where=row_sums > 0)


def choose_ensemble_strategy(
    true_labels: list[int] | np.ndarray,
    model_probabilities: np.ndarray,
    model_weights: list[float] | np.ndarray,
) -> dict:
    from glioma_hc.evaluation.evaluate import compute_multiclass_metrics

    probabilities = np.asarray(model_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1] != len(true_labels):
        raise ValueError("Validation labels and model probability arrays have incompatible shapes.")
    weights = np.asarray(model_weights, dtype=np.float64)
    if weights.shape != (probabilities.shape[0],):
        raise ValueError("Provide one validation-performance weight per model.")

    comparisons = {}
    strategy_order = ("hard_voting", "soft_voting", "weighted_soft_voting")
    for strategy in strategy_order:
        combined = combine_model_probabilities(
            probabilities,
            strategy,
            weights if strategy == "weighted_soft_voting" else None,
        )
        comparisons[strategy] = {
            "metrics": compute_multiclass_metrics(
                np.asarray(true_labels, dtype=int),
                combined,
                SETTINGS.label_order,
            ),
            "model_weights": (
                (weights / weights.sum()).tolist()
                if strategy == "weighted_soft_voting" and weights.sum() > 0
                else None
            ),
        }

    selected = max(
        strategy_order,
        key=lambda strategy: (
            comparisons[strategy]["metrics"]["balanced_accuracy"],
            comparisons[strategy]["metrics"]["macro_f1"],
            comparisons[strategy]["metrics"]["macro_specificity"],
            -strategy_order.index(strategy),
        ),
    )
    return {
        "strategy": selected,
        "selection_split": "val",
        "selection_metric": "balanced_accuracy, then macro_f1, then macro_specificity",
        "comparisons": comparisons,
        "model_order": [],
    }


def choose_deployment_candidate(
    model_metrics: dict[str, dict],
    ensemble_selection: dict,
) -> dict:
    """Choose a single validated model or the validated ensemble without test data."""
    metric_names = ("balanced_accuracy", "macro_f1", "macro_specificity")
    candidates = []
    model_comparisons = {}
    ensemble_comparisons = ensemble_selection.get("comparisons", {})

    for name, metrics in model_metrics.items():
        if all(isinstance(metrics.get(metric), (int, float)) for metric in metric_names):
            model_comparisons[name] = metrics
            candidates.append({
                "kind": "single_model",
                "model": name,
                "strategy": "single_model",
                "metrics": metrics,
            })

    for strategy in ("hard_voting", "soft_voting", "weighted_soft_voting"):
        comparison = ensemble_comparisons.get(strategy, {})
        metrics = comparison.get("metrics", {})
        if all(isinstance(metrics.get(metric), (int, float)) for metric in metric_names):
            candidates.append({
                "kind": "ensemble",
                "model": None,
                "strategy": strategy,
                "metrics": metrics,
            })

    if not candidates:
        raise ValueError("Validation metrics are required to select a deployment candidate.")

    selected = max(
        candidates,
        key=lambda candidate: (
            *(float(candidate["metrics"][metric]) for metric in metric_names),
            -candidates.index(candidate),
        ),
    )
    return {
        "kind": selected["kind"],
        "model": selected["model"],
        "strategy": selected["strategy"],
        "selection_split": "val",
        "selection_metric": "balanced_accuracy, then macro_f1, then macro_specificity",
        "validation_metrics": selected["metrics"],
        "comparisons": {
            "models": model_comparisons,
            "ensemble": ensemble_comparisons,
        },
    }


def deployment_from_training_summary(summary: dict) -> dict | None:
    if not isinstance(summary, dict):
        return None
    deployment = summary.get("deployment")
    if isinstance(deployment, dict) and deployment.get("selection_split") == "val":
        return deployment

    raw_records = summary.get("models", [])
    if not isinstance(raw_records, list):
        raw_records = []
    records = {
        record.get("name"): record
        for record in raw_records
        if isinstance(record, dict)
    }
    model_metrics = {
        name: records[name].get("validation_metrics", {})
        for name in records
        if isinstance(records[name].get("validation_metrics"), dict)
    }
    ensemble = summary.get("ensemble", {})
    if not isinstance(ensemble, dict):
        return None
    if (
        ensemble.get("selection_split") == "val"
        and ensemble.get("strategy")
        in {"hard_voting", "soft_voting", "weighted_soft_voting"}
        and ensemble.get("model_order") == list(REQUIRED_ENSEMBLE_MODELS)
    ):
        return {
            "kind": "ensemble",
            "model": None,
            "strategy": ensemble["strategy"],
            "selection_split": "val",
            "selection_metric": ensemble.get("selection_metric"),
        }
    try:
        return choose_deployment_candidate(model_metrics, ensemble)
    except ValueError:
        if (
            ensemble.get("selection_split") == "val"
            and ensemble.get("strategy")
            in {"hard_voting", "soft_voting", "weighted_soft_voting"}
        ):
            return {
                "kind": "ensemble",
                "model": None,
                "strategy": ensemble["strategy"],
                "selection_split": "val",
            }
        return None
