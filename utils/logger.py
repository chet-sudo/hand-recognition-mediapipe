"""Lightweight logging utilities for experiment metrics."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunMetadata:
    experiment: str
    algorithm: str
    environment: str
    hyperparameters: Dict[str, Any]
    seed: Optional[int]
    timestamp: str


class MetricsLogger:
    """Persist per-episode metrics and run metadata to disk."""

    def __init__(
        self,
        experiment: str,
        algorithm: str,
        environment: str,
        seed: Optional[int],
        hyperparameters: Optional[Dict[str, Any]] = None,
        base_dir: str = "results",
        run_label: Optional[str] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_label = run_label or timestamp
        self.run_dir = Path(base_dir) / experiment / algorithm / f"seed-{seed if seed is not None else 'na'}" / safe_label
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = RunMetadata(
            experiment=experiment,
            algorithm=algorithm,
            environment=environment,
            hyperparameters=hyperparameters or {},
            seed=seed,
            timestamp=timestamp,
        )

        self._csv_path = self.run_dir / "episodes.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["episode", "return", "length"])
        self._csv_writer.writeheader()

        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(asdict(self.metadata), indent=2))

        self.summary: Dict[str, Any] = {}

    def log_episode(self, episode: int, episode_return: float, length: int) -> None:
        self._csv_writer.writerow({"episode": episode, "return": episode_return, "length": length})
        self._csv_file.flush()

    def log_summary(self, **metrics: Any) -> None:
        self.summary.update(metrics)

    def close(self) -> None:
        self._csv_file.close()
        if self.summary:
            summary_path = self.run_dir / "summary.json"
            summary_path.write_text(json.dumps(self.summary, indent=2))

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
