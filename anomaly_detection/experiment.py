"""One controlled experiment."""

from torch.utils.data import Dataset

from .evaluation import AnomalyEvaluator, EvaluationResult
from .model import AnomalyModel


class ExperimentRunner:
    def __init__(self, model: AnomalyModel, evaluator: AnomalyEvaluator) -> None:
        self.model = model
        self.evaluator = evaluator

    def run(self, dataset: Dataset) -> EvaluationResult:
        raise NotImplementedError("The experiment loop comes next")
