r"""
replay.py
----------
Implementation of a Replay Memory buffer using the PyTorch library.
"""

import torch
from metis.base import Replay


class NoReplay(Replay):
    def sample(self):
        out = self._compile(self.buffer)
        self.buffer = []
        return out


class ExperienceReplay(Replay):
    def sample(self, n: int):
        idx = torch.randperm(len(self))[:n].tolist()
        return self._compile([self[i] for i in idx])
