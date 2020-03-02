from copy import deepcopy
from time import sleep
from typing import Callable, Sequence
import random

import gym
import numpy as np
import scipy.signal
import torch
from torch import Tensor, nn
from numpy import ndarray

from metis.agents import Actor


def numpymethod(function: Callable) -> Callable:
    """Decorator function, which automatically casts any `np.ndarray` input
    arguments (or keyword arguments) to `torch.Tensor`.  If *any* numpy arrays
    were provided as inputs, casts all returned multidimensional arrays to
    `np.ndarray` objects.

    With this decorator, users can expect to receive the same *type* of array
    as was provided to the function.

    Examples:
        @torch_method
        def add_noise(x, sigma=1.0):
            return x + sigma * torch.rand(x.shape)
    """

    def new_method(*args, **kwargs):
        args_ = tuple(
            a.data.cpu().numpy() if isinstance(a, Tensor) else a for a in args
        )
        kwargs_ = {
            k: v.data.cpu().numpy() if isinstance(v, Tensor) else v
            for k, v in kwargs.items()
        }
        out = function(*args_, **kwargs_)

        if isinstance(out, ndarray):
            return torch.as_tensor(out, dtype=torch.float)
        elif isinstance(out, tuple):
            return tuple(
                torch.as_tensor(a, dtype=torch.float)
                if isinstance(a, ndarray) else a for a in out
            )
        elif isinstance(out, list):
            return [
                torch.as_tensor(a, dtype=torch.float)
                if isinstance(a, ndarray) else a for a in out
            ]
        else:
            return torch.as_tensor(out)

    return new_method


# def torchmethod(function: Callable) -> Callable:
#     """TODO: Docstring"""
#     def new_method(*args, **kwargs):
#         args_ = tuple(
#             a.data.cpu().numpy() if isinstance(a, Tensor) else a for a in args
#         )
#         kwargs_ = {
#             k: v.data.cpu().numpy() if isinstance(v, Tensor) else v
#             for k, v in kwargs.items()
#         }
#         out = function(*args_, **kwargs_)
#
#         if isinstance(out, ndarray):
#             return torch.as_tensor(out).type(torch.float)
#         elif isinstance(out, tuple):
#             return tuple(
#                 torch.as_tensor(a).type(torch.float)
#                 if isinstance(a, ndarray) else a for a in out
#             )
#         elif isinstance(out, list):
#             return [
#                 torch.as_tensor(a).type(torch.float)
#                 if isinstance(a, ndarray) else a for a in out
#             ]
#         else:
#             return out
#
#     return new_method


def torchenv(env: gym.Env):
    new_env = deepcopy(env)
    new_env.reset = numpymethod(new_env.reset)
    new_env.step = numpymethod(new_env.step)
    new_env.action_space.sample = numpymethod(new_env.action_space.sample)
    new_env.observation_space.sample = numpymethod(new_env.observation_space.sample)

    return new_env


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
    env = torchenv(env)
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


def seed(value: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@numpymethod
def discount_values(raw_values, dones, discount):
    def discount_fn(values):
        out = scipy.signal.lfilter([1], [1, float(-discount)], values[::-1], axis=0)[::-1]
        return np.ascontiguousarray(out)

    if not np.any(dones > 1e-6):
        return discount_fn(raw_values)

    idx = 0
    values = np.zeros_like(raw_values)
    episode_ends = np.nonzero(dones > 1e-6)[0]
    for end in episode_ends:
        if end > idx + 1:
            values[idx:end+1] = discount_fn(raw_values[idx:end+1])
        idx = end + 2
    values[idx:] = discount_fn(raw_values[idx:])

    return np.ascontiguousarray(values)


def compile_tensors(
    inputs: Sequence[Tensor or Sequence[Tensor]],
    device: torch.device = None,
) -> Tensor or Sequence[Tensor]:
    if isinstance(inputs, Tensor):
        out = inputs
    elif isinstance(inputs[0], Tensor):
        out = torch.stack(tuple(inputs), dim=0)
    elif isinstance(inputs[0], Sequence):
        num_outputs = len(inputs[0])
        out = tuple(
            torch.cat([x[i] for x in inputs]) for i in range(num_outputs)
        )
    else:
        out = torch.tensor(inputs, dtype=torch.float)

    if device is not None:
        out = out.to(device)

    return out


def get_device(obj: Tensor or nn.Module or nn.DataParallel):
    if isinstance(obj, Tensor):
        return obj.device
    elif isinstance(obj, nn.Module):
        for p in obj.parameters():
            return p.device
    elif isinstance(obj, nn.DataParallel):
        return obj.output_device
    else:
        raise TypeError(f"Cannot get device for type {type(obj)}")
