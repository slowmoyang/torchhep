import operator
from typing import Callable

# TODO min_delta

class EarlyStopping:
    direction: str
    patience: int
    verbose: bool
    comparator: Callable[[float, float], bool]
    worst_value: float
    wait: int

    allowed_directions = ('minimize', 'maximize')

    def __init__(self,
                 direction: str = 'minimize',
                 worst_value: float | None = None,
                 patience: int = 10,
                 verbose: bool = False,
    ) -> None:
        if direction not in self.allowed_directions:
            raise ValueError(f'{direction=} not in {self.allowed_directions=}')

        self.direction = direction
        self.patience = patience
        self.verbose = verbose

        if worst_value is None:
            if self.direction == 'minimize':
                self.worst_value = float('inf')
            else:
                self.worst_value = float('-inf')
        else:
            self.worst_value = worst_value

        self.comparator = operator.lt if self.direction == 'minimize' else operator.gt
        self.wait = 0

    def step(self, metric: float) -> bool:
        stop = False

        if self.comparator(metric, self.worst_value):
            self.worst_value = metric
            self.wait = 0 # reset
        else:
            self.wait += 1
            if self.wait >= self.patience:
                stop = True

        if self.verbose and self.wait > 0:
            print(f'wait / patience = {self.wait: 5d} / {self.patience: 5d}')
        return stop
