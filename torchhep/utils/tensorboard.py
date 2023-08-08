from functools import cached_property
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import torch
import torch.utils.tensorboard.writer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorBoardWriter(torch.utils.tensorboard.writer.SummaryWriter):
    global_step: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = 0

        self._epoch2step: dict[int, int] = {}

    def add_result(self, phase: str, step: Optional[int] = None, **kwargs) -> None:
        step = step or self.global_step
        for key, value in kwargs.items():
            self.add_scalar(f'{key}/{phase}', value, step)

    def add_train_step_result(self, phase: str = 'training', **kwargs):
        self.global_step += 1
        return self.add_result(phase, **kwargs)

    def add_eval_result(self, phase: str, step: Optional[int] = None, **kwargs):
        return self.add_result(phase, step, **kwargs)

    def add_epoch(self, epoch):
        self.add_scalar('epoch', epoch, self.global_step)
        self._epoch2step[epoch] = self.global_step

    def convert_epoch_to_step(self, epoch: int) -> int:
        return self._epoch2step[epoch]


class TensorBoardReader:
    log_dir: Path

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.event_accumulator = EventAccumulator(str(log_dir)).Reload()

    def keys(self):
        return self.event_accumulator.Tags()

    @cached_property
    def scalars(self) -> dict[str, Any]:
        scalars = self.event_accumulator.scalars

        result = {}
        for key in scalars.Keys():
            inner = result
            *token_list, scalar_name = key.split('/')
            for token in token_list:
                if token not in inner:
                    inner[token] = {}
                inner = inner[token]
            inner[scalar_name] = pd.DataFrame(scalars.Items(key))
        return result


    @classmethod
    def from_summary_writer(cls, summary_writer):
        summary_writer.flush()
        return cls(summary_writer.get_logdir())

    def convert_epoch_to_step(self, epoch: int):
        '''
        return step corresponding to the end of a given epoch
        '''
        for _, row in self.scalars['Epoch'].iterrows():
            if int(row.step) == epoch:
                return row.step
        else:
            raise RuntimeError
