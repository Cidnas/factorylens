"""Experiment artifact storage."""

from pathlib import Path
import json


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def experiment_path(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def save(self, experiment_id: str, result: dict, predictions: list[dict]) -> Path:
        path = self.experiment_path(experiment_id)
        path.mkdir(parents=True, exist_ok=False)
        for name, data in (("result.json", result), ("predictions.json", predictions)):
            with (path / name).open("w") as file:
                json.dump(data, file, indent=2, allow_nan=False)
                file.write("\n")
        return path
