"""
callbacks.py
-------------
Callbacks for use during model training
"""

import torch
from torch import nn


class ModelCheckpoint:
    """Saves the entire model using 'pickle'.  'StateDictCheckpoint' is preferred
    whenever possible, because this creates issues whenever source files change.
    But it's still useful in many scenarios, so we include it here.
    """
    def __init__(self, model: nn.Module, path: str, frequency: int = 1):
        self.model = model
        self.path = path
        self.frequency = frequency

        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter % self.frequency == 0:
            torch.save(self.model, self.path)


class StateDictCheckpoint:
    """Saves the model's state dictionary using 'pickle'.  NOTE: This does not
    implicitly save any of the model source code, as with 'ModelCheckpoint'.
    The resulting state dictionary is much easier to use across projects/libraries.
    However, it requires users to manually ensure that the underlying source code
    does not change.
    """
    def __init__(self, module: nn.Module, path: str, frequency: int = 1):
        self.module = module
        self.path = path
        self.frequency = frequency

        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter % self.frequency == 0:
            torch.save(self.module.state_dict(destination="cpu"), self.path)
