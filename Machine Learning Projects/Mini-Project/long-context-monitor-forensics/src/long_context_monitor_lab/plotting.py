from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_rate_curves(
    x: Sequence[float],
    rates: Mapping[str, Sequence[float]],
    intervals: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
):
    """Plot one curve per method; return the figure and axis."""

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x_values = np.asarray(x, dtype=float)
    for name, values in rates.items():
        y_values = np.asarray(values, dtype=float)
        ax.plot(x_values, y_values, marker="o", label=name)
        if intervals is not None and name in intervals:
            low, high = intervals[name]
            ax.fill_between(x_values, low, high, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_contribution_summary(
    labels: Sequence[str],
    feature_terms: Sequence[float],
    residual_terms: Sequence[float],
    *,
    title: str,
):
    """Plot aggregate selected-feature and residual contributions."""

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(len(labels))
    ax.bar(positions, feature_terms, label="selected SAE features")
    ax.bar(
        positions,
        residual_terms,
        bottom=feature_terms,
        label="SAE residual",
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("mean probe-logit contribution")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def save_figure(fig, path: str | Path) -> None:
    """Create the parent directory and save a tightly cropped figure."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=180)
