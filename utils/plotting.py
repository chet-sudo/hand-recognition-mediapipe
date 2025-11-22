"""Plotting utilities for comparing algorithms."""
from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt


def plot_learning_curves(
    results: Dict[str, List[float]],
    std_results: Optional[Dict[str, List[float]]] = None,
    evaluation_scores: Optional[Dict[str, float]] = None,
    window: int = 10,
    title: str = "Learning Curves",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot episode returns for multiple agents on the same axes.

    The default behavior saves plots to ``plots/<title>.png`` so training
    scripts do not block on GUI windows in headless environments. Set
    ``show=True`` to display the figure interactively instead.
    """
    plt.figure(figsize=(8, 4))
    for label, returns in results.items():
        if len(returns) == 0:
            continue
        smoothed = _moving_average(returns, window)
        x_axis = list(range(1, len(smoothed) + 1))
        line, = plt.plot(x_axis, smoothed, label=f"{label} (train)")
        if std_results and label in std_results:
            std_curve = std_results[label]
            upper = [m + s for m, s in zip(smoothed, std_curve)]
            lower = [m - s for m, s in zip(smoothed, std_curve)]
            plt.fill_between(x_axis, lower, upper, color=line.get_color(), alpha=0.15)
        if evaluation_scores and label in evaluation_scores:
            plt.axhline(
                evaluation_scores[label],
                linestyle="--",
                color=line.get_color(),
                alpha=0.8,
                label=f"{label} eval",
            )
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return (smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = Path(save_path) if save_path else Path("plots") / f"{title.lower().replace(' ', '_')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    averaged = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        averaged.append(sum(values[start : i + 1]) / (i - start + 1))
    return averaged
