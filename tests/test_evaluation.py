import numpy as np
import pytest

from glioma_hc.evaluation.artifacts import (
    save_confusion_matrix_figure,
    save_evaluation_figures,
    summarize_prediction_confidence,
)
from glioma_hc.evaluation.evaluate import (
    compute_metrics,
    compute_multiclass_metrics,
    write_summary,
)


def test_compute_metrics_uses_binary_positive_probability():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])

    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["cohen_kappa"] == 1.0


def test_compute_multiclass_metrics_reports_per_class_scores():
    metrics = compute_multiclass_metrics(
        [0, 1, 2, 3],
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.05, 0.1, 0.8, 0.05],
            [0.05, 0.1, 0.1, 0.75],
        ],
        ["glioma", "meningioma", "notumor", "pituitary"],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_specificity"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["per_class"]["pituitary"]["auc"] == 1.0
    assert metrics["per_class"]["glioma"]["sensitivity"] == 1.0
    assert metrics["per_class"]["glioma"]["specificity"] == 1.0
    assert metrics["per_class"]["glioma"]["support"] == 1
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_compute_multiclass_metrics_rejects_probability_shape_mismatch():
    with pytest.raises(ValueError, match="one column per configured class"):
        compute_multiclass_metrics([0], [[0.4, 0.6]], ["glioma", "meningioma", "notumor", "pituitary"])


def test_compute_multiclass_metrics_calculates_one_vs_rest_specificity():
    metrics = compute_multiclass_metrics(
        [0, 0, 1, 1, 2, 2, 3, 3],
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
            [0.02, 0.03, 0.9, 0.05],
            [0.02, 0.03, 0.1, 0.85],
            [0.05, 0.05, 0.05, 0.85],
            [0.9, 0.03, 0.03, 0.04],
        ],
        ["glioma", "meningioma", "notumor", "pituitary"],
    )

    assert metrics["per_class"]["glioma"]["support"] == 2
    assert metrics["per_class"]["glioma"]["sensitivity"] == 0.5
    assert metrics["per_class"]["glioma"]["specificity"] == pytest.approx(5 / 6)


def test_evaluation_artifacts_write_confusion_and_roc_figures(tmp_path):
    labels = [0, 1, 2, 3]
    probabilities = [
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.1, 0.1],
        [0.05, 0.1, 0.8, 0.05],
        [0.05, 0.1, 0.1, 0.75],
    ]

    paths = save_evaluation_figures(
        np.asarray(labels),
        np.asarray(probabilities),
        ["glioma", "meningioma", "notumor", "pituitary"],
        tmp_path,
        "test",
    )

    assert set(paths) == {"confusion_matrix", "roc_curves"}
    assert all((tmp_path / path).is_file() for path in paths.values())
    confidence = summarize_prediction_confidence(np.asarray(probabilities))
    assert confidence["samples"] == 4
    assert confidence["mean"] == pytest.approx(0.7625)
    assert confidence["minimum"] == pytest.approx(0.7)


def test_write_summary_backs_up_existing_report(tmp_path):
    output = tmp_path / "final_test_results.json"
    output.write_text('{"accuracy": 0.8}', encoding="utf-8")

    write_summary({"accuracy": 0.9}, output)

    assert (tmp_path / "final_test_results_backup.json").read_text(encoding="utf-8") == (
        '{"accuracy": 0.8}'
    )
    assert '"accuracy": 0.9' in output.read_text(encoding="utf-8")


def test_confusion_figure_accepts_saved_counts_without_prediction_replay(tmp_path):
    path = tmp_path / "confusion.png"

    result = save_confusion_matrix_figure(
        [[3, 1], [0, 4]],
        ["glioma", "meningioma"],
        path,
    )

    assert result == str(path)
    assert path.is_file()