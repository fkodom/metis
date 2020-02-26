from abc import ABC, abstractmethod
from typing import Sequence
from time import sleep

import gym
from torch import nn, Tensor

from metis import utils

State = Tensor or Sequence[Tensor]
Action = Tensor or float or int
LogProb = Tensor or Sequence[Tensor]
Value = Tensor
Observation = Sequence[Tensor or float or int or bool]
Batch = Sequence[Tensor or Sequence[Tensor]]


class Actor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: State) -> (Action, LogProb):
        r"""Computes action probabilities for the given environment state."""


class Critic(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: State, action: Action) -> Value:
        r"""Pushes the environment state through the network, and returns action
        probabilities and state value.
        """


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
    def _compile(samples: Sequence[Observation]):
        nitems = len(samples[0])
        return tuple(
            utils.compile_tensors([s[i] for s in samples]) for i in range(nitems)
        )

    def append(self, observation: Observation):
        self.buffer.append(observation)

    @abstractmethod
    def sample(self, *args, **kwargs) -> Batch:
        """"""


def play(
    env: gym.Env,
    actor: Actor,
    max_turns: int = 1000,
    frame_rate: float = 9e9,
    return_frames: bool = False,
) -> list or None:
    r"""Plays one game (episode), and visualizes the game environment.

    :param max_turns: Maximum number of turns (or frames) to play in one game
    :param frame_rate: If render = True, controls the frame rate (in frames/sec) of the episode
    """
    env = utils.torchenv(env)
    state = env.reset()
    frames = []

    for turn in range(max_turns):
        sleep(1 / (frame_rate + 1e-6))
        env.render(mode='human')
        if return_frames:
            frames.append(env.render('rgb_array'))

        action = actor.act(state)
        state, _, done, _ = env.step(action)
        if done:
            break

    env.close()
    if return_frames:
        return frames
