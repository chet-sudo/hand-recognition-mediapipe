from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def resolve_seeds(seeds: Iterable[int] | int | None, seed_mode: str = "list") -> List[int]:
    """Normalize seed input into a list of integers.

    Args:
        seeds: Either an iterable of seeds, a single integer, or ``None``.
        seed_mode: When ``"count"``, a single integer value will be expanded to
            ``range(value)`` to support running multiple seeds by count.
    """

    if seeds is None:
        return [0]
    if isinstance(seeds, int):
        expanded = list(range(seeds)) if seed_mode == "count" else [seeds]
        return expanded or [0]

    seeds_list = list(seeds)
    if len(seeds_list) == 1 and seed_mode == "count":
        expanded = list(range(seeds_list[0]))
        return expanded or [0]
    return seeds_list or [0]


def aggregate_returns(runs: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Compute mean and standard deviation across multiple return sequences."""

    array = np.array([np.array(r) for r in runs])
    return array.mean(axis=0).tolist(), array.std(axis=0).tolist()


def write_experiment_summary(
    experiment_label: str,
    mean_returns: Dict[str, List[float]],
    std_returns: Dict[str, List[float]],
    eval_means: Dict[str, float],
    eval_stds: Dict[str, float],
) -> Path:
    """Persist aggregated metrics for quick inspection."""

    output_dir = Path("results") / experiment_label
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mean_returns": mean_returns,
        "std_returns": std_returns,
        "evaluation_mean": eval_means,
        "evaluation_std": eval_stds,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path
