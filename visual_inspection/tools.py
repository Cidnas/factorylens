"""Small tool surface exposed to the research agent."""

from collections.abc import Callable

from .experiment import ExperimentRunner


def experiment_tools(runner: ExperimentRunner) -> tuple[Callable[..., str], ...]:
    def run_experiment(category: str) -> str:
        """Run one approved experiment for an MVTec category."""

        raise NotImplementedError("Experiment execution comes next")

    return (run_experiment,)
