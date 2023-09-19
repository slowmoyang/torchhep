import abc
from typing import Any
import torch
import optuna


class ObjectiveBase(abc.ABC):

    def __init__(self, num_epochs: int) -> None:
        self.num_epochs = num_epochs

    @abc.abstractmethod
    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    def train(self, suggestion: dict[str, Any]):
        ...

    @abc.abstractmethod
    def validate(self, suggestion: dict[str, Any]) -> Any:
        ...

    def run(self, trial: optuna.Trial): # FIXME rename
        try:
            suggestion = self.suggest(trial)
            for epoch in range(self.num_epochs):
                self.train(suggestion)
                val_result = self.validate(suggestion)
                metric = val_result[self.target_name]
                trial.report(metric, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return metric
        except torch.cuda.OutOfMemoryError as error: # type: ignore
            print(error)
            raise optuna.exceptions.TrialPruned() # FIXME

    def __call__(self, trial: optuna.Trial):
        return self.run(trial)

    @classmethod
    @property
    @abc.abstractmethod
    def target_name(cls) -> str:
        ...

    @classmethod
    @property
    @abc.abstractmethod
    def direction(cls) -> str:
        ...
