"""Anomaly-detection evaluation boundary."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class EvaluationResult:
    image_auroc: float
    pixel_auroc: float
    aupro: float


class AnomalyEvaluator:
    def evaluate(self, predictions: Tensor, targets: Tensor) -> EvaluationResult:
        raise NotImplementedError("Metric calculation comes next")
