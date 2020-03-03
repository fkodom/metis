r"""
replay.py
----------
Replay buffers for sampling past experience during RL training.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor

from metis import utils


# Data types, for better readability
Observation = Sequence[Tensor or float or int or bool]
Batch = Sequence[Tensor or Sequence[Tensor]]


class Replay(ABC):
    """Base class for all replay buffers.  Nearly all public and private methods
    for common replay buffers are defined here -- really only the 'sample' method
    needs to be defined for subclasses.

    NOTE:
        * The buffer itself is implemented as a simple Python list.  This feels
          like the most generic, flexible way to do it.  We also don't take much
          of a performance hit for doing this, as opposed to pre-allocating the
          buffer memory with Tensor objects.  Especially for more complex
          environments, the performance difference will be negligible when
          compared to time spent simulating the environment, pushing data to/from
          GPU, and computing action probabilities.
        * Replay is completely agnostic to the data that goes into it.  That is
          to say, it accepts most any data type (e.g. Tensor, list, tuple, float,
          bool, etc.) and compiles them back into Tensor objects in the 'sample'
          method.  States/actions can also contain *multiple* Tensors (for
          environments with multiple control/state inputs).  The goal is to make
          Replay as flexible as possible.
    """
    def __init__(self, maxsize: int = int(1e4)):
        """
        Parameters
        ----------
        maxsize: (int) Maximum number of experiences that can be stored in the
            replay buffer.  Once this number is reached, old experiences are
            incrementally removed from the buffer.
        """
        self.maxsize = int(maxsize)
        self.buffer: list = []

    def __repr__(self):
        return f"{self.__class__.__name__}(maxsize={self.maxsize})"

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer.__getitem__(item)

    @staticmethod
    def _compile(samples: Sequence[Observation], device: torch.device = None):
        """Compiles all sampled observations into Tensor objects for training
        RL agents.  This is mostly just convenience method, which wraps
        'metis.utils.compile_tensors' to make subclass implementations simpler.

        Parameters
        ----------
        samples: (Sequence[Observation]) Collection of raw samples taken from
            the memory buffer, which need to be compiled into Tensors.
        device: (torch.device, optional) If specified, this method will try to
            push all resulting Tensors to that device.  Otherwise, it leaves
            all Tensors on their current devices.
        """
        return tuple(
            utils.compile_tensors([s[i] for s in samples], device=device)
            for i in range(len(samples[0]))
        )

    def append(self, observation: Observation):
        """Appends an observation (experience) to the replay buffer.  If the
        buffer exceeds its maximum length, then it removes the oldest samples
        from the beginning of the buffer.

        Parameters
        ----------
        observation

        Returns
        -------

        """
        self.buffer.append([torch.as_tensor(x).detach().cpu() for x in observation])
        if len(self.buffer) > self.maxsize:
            self.buffer.pop(0)

    @abstractmethod
    def sample(self, *args, device: torch.device = None, **kwargs) -> Batch:
        """Draws samples from the memory buffer, compiles them into Tensors,
        and optionally pushes them to CPU/GPU for training RL agents.
        """


class NoReplay(Replay):
    """Replay object for on-policy training, where experiences are only sampled
    once before training agents and completely emptying the buffer.  (I.e. there
    is no actual *replay* -- samples are only used once.)
    """
    def sample(self, device: torch.device = None, **kwargs):
        """Draws *all* samples from the memory buffer, compiles them into
        Tensors, and optionally pushes them to CPU/GPU for training RL agents.
        Then, empties the replay buffer.
        """
        out = self._compile(self.buffer, device=device)
        self.buffer = []
        return out


class ExperienceReplay(Replay):
    """Basic experience replay, which stores large numbers of past experiences,
    and samples randomly from them for training.
    """
    # noinspection PyMethodOverriding
    def sample(self, n: int, device: torch.device = None, **kwargs):
        """Draws *random* samples from the memory buffer, compiles them into
        Tensors, and optionally pushes them to CPU/GPU for training RL agents.

        Parameters
        ----------

        """
        idx = torch.randperm(len(self))[:n].tolist()
        return self._compile([self[i] for i in idx], device=device)
