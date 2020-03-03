"""
metis/utils.py
--------------
Utility functions for training and visualizing RL agents in PyTorch.
"""

from copy import deepcopy
from time import sleep
from typing import Callable, Sequence
import random

import gym
import torch
from torch import Tensor, nn
import numpy as np
from scipy.signal import lfilter
from numpy import ndarray

from metis.agents import Actor


def numpymethod(function: Callable) -> Callable:
    """Decorator function, which automatically casts any torch Tensor input
    arguments (both positional and keyword) to numpy arrays.  Then, casts all
    numpy arrays back to Tensors before returning.  Essentially, this makes it
    easier to use Numpy functions (both built-in and custom) inter-mixed with
    PyTorch code.

    NOTE:  This function performs a shallow search for ndarray/Tensor arguments,
    and will not detect deeply nested ndarray/Tensor values (i.e. from
    nested dictionaries, lists, tuples, etc.).

    Parameters
    ----------
    function: (Callable) Numpy function to decorate for use with Torch code

    Examples
    --------
    @numpymethod
    def add_noise(x, sigma=1.0):
        return x + sigma * np.random.randn(x.shape)
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


def torchmethod(function: Callable) -> Callable:
    """Decorator function, which automatically casts any numpy array input
    arguments (both positional and keyword) to torch Tensors.  Then, casts all
    Tensors back to numpy arrays before returning.  Essentially, this makes it
    easier to use torch functions (both built-in and custom) inter-mixed with
    Numpy code.

    NOTE:  This function performs a shallow search for ndarray/Tensor arguments,
    and will not detect deeply nested ndarray/Tensor values (i.e. from
    nested dictionaries, lists, tuples, etc.).

    Parameters
    ----------
    function: (Callable) Torch function to decorate for use with Numpy code

    Examples
    --------
    @torchmethod
    def add_noise(x, sigma=1.0):
        return x + sigma * torch.randn(x.shape)
    """

    def new_method(*args, **kwargs):
        args_ = tuple(
            torch.as_tensor(a) if isinstance(a, ndarray) else a for a in args
        )
        kwargs_ = {
            k: torch.as_tensor(v) if isinstance(v, ndarray) else v
            for k, v in kwargs.items()
        }
        out = function(*args_, **kwargs_)

        if isinstance(out, Tensor):
            return out.detach().cpu().numpy()
        elif isinstance(out, tuple):
            return tuple(
                a.detach().cpu().numpy()
                if isinstance(a, Tensor) else a for a in out
            )
        elif isinstance(out, list):
            return [
                a.detach().cpu().numpy()
                if isinstance(a, Tensor) else a for a in out
            ]
        else:
            return out

    return new_method


def torchenv(env: gym.Env):
    """Decorator function, which makes OpenAI Gym environments return torch
    Tensors instead of numpy arrays.  Allows us to integrate with torch more
    easily, and without constantly writing things like:
        state = torch.as_tensor(state, dtype=torch.float)

    Parameters
    ----------
    env: (gym.Env) Gym environment to wrap

    Returns
    -------
    gym.Env: Decorated Gym environment
    """
    new_env = deepcopy(env)
    new_env.reset = numpymethod(new_env.reset)
    new_env.step = numpymethod(new_env.step)
    new_env.action_space.sample = numpymethod(new_env.action_space.sample)
    new_env.observation_space.sample = numpymethod(new_env.observation_space.sample)

    return new_env


def play(
    env: gym.Env,
    actor: Actor,
    max_ep_len: int = 1000,
    frame_rate: float = 9e9,
    return_frames: bool = False,
) -> list:
    """Wraps the Gym environment using 'torchenv', plays one episode (game),
    and visualizes the environment.  Also provides the option to return a list
    of raw frames from the episode (intended for saving videos and/or GIFs, but
    use it however you want).

    Parameters
    ----------
    env: (gym.Env) Gym environment to play an episode
    actor: (Actor) Actor (policy) network for choosing actions for each state.
    max_ep_len: (int, optional) Maximum number of steps per episode.
        Default: 1000.
    frame_rate: (float, optional) Maximum frame rate for visualization.  In many
        cases, the actual frame rate will be determined by the computational
        speed of the actor and environment.  Default: 9e9 (max speed).
    return_frames: (bool, optional) If True, this function returns a list of
        all raw frame arrays, for saving videos of episodes.  Default: False.

    Returns
    -------
    list: If return_frames is True, returns a list of all raw frame arrays.
        Otherwise, returns None.
    """
    env = torchenv(env)
    state = env.reset()
    frames = []

    for turn in range(max_ep_len):
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


# noinspection PyUnresolvedReferences
def seed(value: int):
    """Convenience method for seeding random number generators across torch
    and numpy.  Also sets CUDNN backend to deterministic, and turns off
    benchmarking mode (both introduce indeterminate behavior).

    Parameters
    ----------
    value: (int) Seed value used for all random number generators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@numpymethod
def discount_values(raw_values: ndarray, dones: ndarray, discount: float) -> ndarray:
    """Function for discounting rewards from past experience.  We use the
    'scipy.signal.lfilter' for now, in order to match OpenAI's 'spinup' code.
    Hopefully, we can replace this with PyTorch-only code in the near future.
    (Would prevent the need to convert to/from Numpy arrays all the time.)

    Parameters
    ----------
    raw_values: (ndarray) Original reward/advantage values to discount
    dones: (ndarray) Array of 'done' signals, indicating whether the episode
        ended on that particular turn
    discount: (float) Multiplicative factor for discounting rewards.  Commonly
        referred to as 'gamma', but in general applications this could be
        referring to several different values.

    Returns
    -------
    ndarray: Discounted reward values
    """
    def discount_fn(vals: ndarray) -> ndarray:
        out = lfilter([1], [1, float(-discount)], vals[::-1], axis=0)[::-1]
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
    inputs: Sequence[Tensor or Sequence[Tensor]], device: torch.device = None,
) -> Tensor or Sequence[Tensor]:
    """Convenience method for compiling a sequence of values as a Tensor (or
    Sequence of Tensors), and pushing it to the specified device.  We do that
    over and over again for Experience Replay objects, so this method greatly
    simplifies our implementation of generic replays.

    Parameters
    ----------
    inputs: (Sequence[Tensor or Sequence[Tensor]]) Values to compile into
        Tensor object(s).
    device: (torch.device, optional) If specified, pushes all values to this
        device.  Otherwise, leaves them on the current device.

    Returns
    -------
    Tensor or Sequence[Tensor]:  Compiled Tensor(s)
    """
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


def get_device(obj: Tensor or nn.Module or nn.DataParallel) -> torch.device:
    """Convenience method for getting the device ID for PyTorch objects.
    Specifically, this helps get the device for 'nn.Module' objects.

    Parameters
    ----------
    obj: (PyTorch object) Object to get the device for.

    Returns
    -------
    torch.device: Device where the object is located.
    """
    if isinstance(obj, Tensor):
        return obj.device
    elif isinstance(obj, nn.Module):
        for p in obj.parameters():
            return p.device
    elif isinstance(obj, nn.DataParallel):
        return obj.output_device
    else:
        raise TypeError(f"Could not get device for type {type(obj)}")
