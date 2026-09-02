from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

import numpy as np


def confusion_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Return counts, TPR, FPR, and balanced accuracy."""

    predictions = scores >= threshold

    positive = labels == 1
    negative = labels == 0

    tp = int(np.sum(predictions & positive))
    fn = int(np.sum(~predictions & positive))
    fp = int(np.sum(predictions & negative))
    tn = int(np.sum(~predictions & negative))

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    tnr = tn / (tn + fp)
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "balanced_accuracy": float(balanced_accuracy),
    }


def grouped_confusion_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[Hashable],
    threshold: float,
) -> dict[Hashable, dict[str, float | int]]:
    """Apply confusion_rates independently within each group."""

    groups_array = np.asarray(groups, dtype=object)

    results = {}

    for group in dict.fromkeys(groups):
        mask = groups_array == group

        results[group] = confusion_rates(
            scores=scores[mask],
            labels=labels[mask],
            threshold=threshold,
        )

    return results


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC as the positive-negative pair ordering rate."""

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    comparisons = positive_scores[:, None] - negative_scores[None, :]

    wins = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)

    total_pairs = comparisons.size

    return float((wins + 0.5 * ties) / total_pairs)


def iid_max_fpr(
    single_opportunity_fpr: float,
    n_opportunities: int | np.ndarray,
) -> float | np.ndarray:
    """Predict max-pooled FPR under independent opportunities."""

    result = 1.0 - (1.0 - single_opportunity_fpr) ** n_opportunities

    if np.ndim(result) == 0:
        return float(result)

    return result


def empirical_max_fpr(
    isolated_negative_scores: np.ndarray,
    threshold: float,
    n_opportunities: int,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """Resample isolated scores into synthetic max-pooled contexts."""

    sampled_scores = rng.choice(
        isolated_negative_scores,
        size=(n_trials, n_opportunities),
        replace=True,
    )

    max_scores = sampled_scores.max(axis=1)

    false_positives = max_scores >= threshold

    return float(false_positives.mean())


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    repetitions: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Example-level percentile bootstrap estimate and interval."""

    estimate = float(statistic(values))

    bootstrap_statistics = np.empty(repetitions, dtype=float)

    n_values = len(values)

    for i in range(repetitions):
        indices = rng.integers(
            0,
            n_values,
            size=n_values,
        )

        resampled_values = values[indices]

        bootstrap_statistics[i] = statistic(resampled_values)

    alpha = 1.0 - confidence

    lower = np.quantile(
        bootstrap_statistics,
        alpha / 2.0,
    )

    upper = np.quantile(
        bootstrap_statistics,
        1.0 - alpha / 2.0,
    )

    return (
        estimate,
        float(lower),
        float(upper),
    )
