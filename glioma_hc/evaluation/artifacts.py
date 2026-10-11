from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve


def summarize_prediction_confidence(probabilities: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Confidence summary requires a non-empty probability matrix.")
    confidence = np.max(values, axis=1)
    return {
        "samples": int(confidence.size),
        "mean": float(np.mean(confidence)),
        "median": float(np.median(confidence)),
        "standard_deviation": float(np.std(confidence)),
        "minimum": float(np.min(confidence)),
        "maximum": float(np.max(confidence)),
        "percentile_10": float(np.percentile(confidence, 10)),
        "percentile_90": float(np.percentile(confidence, 90)),
    }


def save_evaluation_figures(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_dir: str | Path,
    filename_prefix: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != (len(labels), len(class_names)):
        raise ValueError("Evaluation probabilities and class names have incompatible shapes.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "confusion_matrix": destination / f"{filename_prefix}_confusion_matrix.png",
        "roc_curves": destination / f"{filename_prefix}_roc_curves.png",
    }

    matrix = confusion_matrix(
        labels,
        np.argmax(scores, axis=1),
        labels=list(range(len(class_names))),
    )
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.savefig(paths["confusion_matrix"], dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plotted = False
    for class_index, class_name in enumerate(class_names):
        binary_labels = (labels == class_index).astype(int)
        if np.unique(binary_labels).size != 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels,
            scores[:, class_index],
        )
        class_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{class_name} (AUC={class_auc:.3f})",
        )
        plotted = True
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="One-vs-rest ROC curves",
    )
    if plotted:
        axis.legend(loc="lower right")
    else:
        axis.text(0.5, 0.5, "ROC curves unavailable for this split", ha="center")
    figure.savefig(paths["roc_curves"], dpi=160)
    plt.close(figure)
    return {name: str(path) for name, path in paths.items()}


def save_confusion_matrix_figure(
    confusion: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix must have one row and column per class.")
    if np.any(matrix < 0):
        raise ValueError("Confusion matrix counts must be non-negative.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Weighted soft-voting ensemble confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def relative_figure_paths(
    figures: dict[str, str],
    report_directory: str | Path,
) -> dict[str, str]:
    root = Path(report_directory).resolve()
    result: dict[str, str] = {}
    for name, raw_path in figures.items():
        path = Path(raw_path).resolve()
        try:
            result[name] = str(path.relative_to(root))
        except ValueError as exc:
            raise ValueError("Evaluation figure paths must be inside the report directory.") from exc
    return result
