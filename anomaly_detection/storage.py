"""Experiment artifact storage."""

from pathlib import Path


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def experiment_path(self, experiment_id: str) -> Path:
        return self.root / experiment_id
