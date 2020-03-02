r"""
replay.py
----------
Implementation of a Replay Memory buffer using the PyTorch library.
"""

import torch
from metis.base import Replay


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
