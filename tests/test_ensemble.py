import numpy as np
import pytest

from glioma_hc.config import SETTINGS
from glioma_hc.training.ensemble import (
    choose_deployment_candidate,
    choose_ensemble_strategy,
    combine_model_probabilities,
    deployment_from_training_summary,
)
from glioma_hc.training.model_zoo import REQUIRED_ENSEMBLE_MODELS


def test_all_voting_strategies_return_normalized_class_probabilities():
    probabilities = np.asarray([
        [[0.7, 0.1, 0.1, 0.1]],
        [[0.2, 0.6, 0.1, 0.1]],
    ])

    hard = combine_model_probabilities(probabilities, "hard_voting")
    soft = combine_model_probabilities(probabilities, "soft_voting")
    weighted = combine_model_probabilities(
        probabilities,
        "weighted_soft_voting",
        [0.9, 0.1],
    )

    assert hard.shape == soft.shape == weighted.shape == (1, len(SETTINGS.label_order))
    assert np.allclose(hard.sum(axis=1), 1)
    assert soft[0].argmax() == SETTINGS.label_order.index("glioma")
    assert weighted[0].argmax() == SETTINGS.label_order.index("glioma")


def test_strategy_selection_uses_only_validation_scores_and_compares_all_methods():
    labels = np.arange(len(SETTINGS.label_order))
    one_model = np.eye(len(SETTINGS.label_order), dtype=np.float64) * 0.9 + 0.025
    predictions = np.repeat(one_model[None, :, :], 8, axis=0)

    selection = choose_ensemble_strategy(labels, predictions, [0.9] * 8)

    assert selection["selection_split"] == "val"
    assert selection["strategy"] == "hard_voting"
    assert set(selection["comparisons"]) == {
        "hard_voting",
        "soft_voting",
        "weighted_soft_voting",
    }
    assert selection["comparisons"]["hard_voting"]["metrics"]["accuracy"] == 1.0


def test_ensemble_rejects_invalid_probabilities_and_weights():
    with pytest.raises(ValueError, match="sum to one"):
        combine_model_probabilities(np.ones((2, 1, 4)), "soft_voting")
    with pytest.raises(ValueError, match="one weight per model"):
        combine_model_probabilities(
            np.full((2, 1, 4), 0.25),
            "weighted_soft_voting",
            [1.0],
        )


def test_deployment_selection_prefers_strongest_validation_model_over_ensemble():
    model_metrics = {
        "resnet50": {
            "balanced_accuracy": 0.84,
            "macro_f1": 0.83,
            "macro_specificity": 0.94,
        },
        "resnet101": {
            "balanced_accuracy": 0.87,
            "macro_f1": 0.86,
            "macro_specificity": 0.95,
        },
    }
    ensemble = {
        "comparisons": {
            "soft_voting": {
                "metrics": {
                    "balanced_accuracy": 0.86,
                    "macro_f1": 0.85,
                    "macro_specificity": 0.95,
                },
            },
        },
    }

    selection = choose_deployment_candidate(model_metrics, ensemble)

    assert selection["selection_split"] == "val"
    assert selection["kind"] == "single_model"
    assert selection["model"] == "resnet101"
    assert selection["validation_metrics"] == model_metrics["resnet101"]


def test_recovered_summary_honors_its_validation_selected_voting_strategy():
    summary = {
        "ensemble": {
            "strategy": "weighted_soft_voting",
            "selection_split": "val",
            "model_order": list(REQUIRED_ENSEMBLE_MODELS),
        },
    }

    deployment = deployment_from_training_summary(summary)

    assert deployment == {
        "kind": "ensemble",
        "model": None,
        "strategy": "weighted_soft_voting",
        "selection_split": "val",
        "selection_metric": None,
    }
