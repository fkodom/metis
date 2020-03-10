"""
callbacks.py
-------------
Callbacks for use during model training
"""

import torch
from torch import nn


class _Checkpoint:
    def __init__(self, module: nn.Module, path: str, frequency: int = 1):
        self.module = module
        self.path = path
        self.frequency = frequency
        self.counter = 0


class ModelCheckpoint(_Checkpoint):
    """Saves the entire model using 'pickle'.  'StateDictCheckpoint' is preferred
    whenever possible, because this creates issues whenever source files change.
    But it's still useful in many scenarios, so we include it here.
    """
    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.frequency == 0:
            torch.save(self.module, self.path)


class StateDictCheckpoint(_Checkpoint):
    """Saves the model's state dictionary using 'pickle'.  NOTE: This does not
    implicitly save any of the model source code, as with 'ModelCheckpoint'.
    The resulting state dictionary is much easier to use across projects/libraries.
    However, it requires users to manually ensure that the underlying source code
    does not change.
    """
    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.frequency == 0:
            torch.save(self.module.state_dict(), self.path)
