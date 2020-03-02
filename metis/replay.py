r"""
replay.py
----------
Implementation of a Replay Memory buffer using the PyTorch library.
"""
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor

from metis import utils


Observation = Sequence[Tensor or float or int or bool]
Batch = Sequence[Tensor or Sequence[Tensor]]


class Replay(ABC):
    def __init__(self, maxsize: int = int(1e4)):
        self.maxsize = int(maxsize)
        self.buffer = []

    def __repr__(self):
        return f"{self.__class__.__name__}(maxsize={self.maxsize})"

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer.__getitem__(item)

    @staticmethod
    def _compile(samples: Sequence[Observation], device: torch.device = None):
        nitems = len(samples[0])
        return tuple(
            utils.compile_tensors([s[i] for s in samples], device=device)
            for i in range(nitems)
        )

    def append(self, observation: Observation):
        self.buffer.append([torch.as_tensor(x).detach().cpu() for x in observation])

    @abstractmethod
    def sample(self, *args, device: torch.device = None, **kwargs) -> Batch:
        """"""


class NoReplay(Replay):
    def sample(self, device: torch.device = None, **kwargs):
        out = self._compile(self.buffer, device=device)
        self.buffer = []
        return out


class ExperienceReplay(Replay):
    # noinspection PyMethodOverriding
    def sample(self, n: int, device: torch.device = None, **kwargs):
        idx = torch.randperm(len(self))[:n].tolist()
        return self._compile([self[i] for i in idx], device=device)
