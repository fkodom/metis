"""
metis/agents.py
---------------
Common types of Actor/Critic networks for reinforcement learning.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Callable
from math import log

import gym
import torch
from torch import Tensor, nn
import torch.nn.functional as f
from torch.distributions import Normal, Categorical

# Max/min values for clipping log probabilities
LOG_STD_MIN = -20
LOG_STD_MAX = 2


# Common data types for actor/critic modules -- for better readability
State = Tensor or Sequence[Tensor]
Action = Tensor or float or int
LogProb = Tensor or Sequence[Tensor]
Value = Tensor


def mlp(
    sizes: Sequence[int],
    activation: Callable,
    output_activation: Callable = nn.Identity(),
) -> nn.Module:
    """Convenience method for creating a multi-layer perceptrons in PyTorch.
    Very commonly used as the backbone for RL agents, especially when the
    environment does not return raw pixel arrays.

    Parameters
    ----------
    sizes: (Sequence[int]) Sizes of the MLP linear layers.  The first value should
        be the size of the input array.
    activation: (Callable) Activation function applied to the output of each layer
    output_activation: (Callable, optional) Activation function applied to the
        final output of the multi-layer perceptron.  Default: nn.Identity()
        (i.e. no activation).  Softmax is often used for discrete action spaces.

    Returns
    -------
    nn.Module: Multi-layer perception (AKA feedforward neural network)
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


# ----------------------- Standardized *ACTOR* Modules -----------------------


class Actor(nn.Module, ABC):
    """Base class for all Actor networks.  Really, this doesn't impose any
    additional abstract methods compared to 'nn.Module', but we define the return
    types for the network here as well.

    NOTE:
        * The 'forward' method should always return two values: (1) predicted
          action, and (2) logarithmic action probability.  For deterministic actors
          (as in DDPG, TD3), simply replace the log probability with 'None'.
        * For discrete action spaces, we typically return an array of action log
          probabilities -- one for each possible action.  This helps for training
          models with target networks (e.g. SAC, TD3).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: State) -> (Action, LogProb):
        """Computes action and log action probability for the given state."""


class CategoricalActor(Actor):
    """Actor with a categorical (discrete) action space.

    NOTE:
        * We return an array of action log probabilities -- one for each
          possible action.  This helps for training models with target
          networks (e.g. SAC, TD3).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.Tanh(),
    ):
        """
        Parameters
        ----------
        state_dim: (int) Size of the state space
        action_dim: (int) Size of the action space
        hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.
            The first value should be the size of the input array.
        activation: (Callable) Activation function applied to the output of
            each layer
        """
        super().__init__()
        self.logits = mlp([state_dim, *hidden_sizes, action_dim], activation=activation)

    def forward(self, state: State, action: Action = None) -> (Action, LogProb):
        probs = torch.softmax(self.logits(state), dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()

        return action, probs.log()


class GaussianActor(Actor):
    """Actor with a continuous action space, where actions are modeled as a
    Gaussian process.  The policy is stochastic, and the actor network simply
    predicts the mean and standard deviation of the Gaussian distribution to
    sample from.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.Tanh(),
        output_activation: Callable = nn.Identity(),
        action_limit: float = 1.0,
    ):
        """
        Parameters
        ----------
        state_dim: (int) Size of the state space
        action_dim: (int) Size of the action space
        hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.
            The first value should be the size of the input array.
        activation: (Callable) Activation function applied to the output of
            each layer
        output_activation: (Callable, optional) Activation function applied to the
            final output of the multi-layer perceptron.  Default: nn.Identity().
        action_limit: (float, optional) Scales the range of actions returned by
            the agent.  Default: 1.0.
        """
        super().__init__()
        self.action_limit = action_limit
        self.mu = mlp(
            [state_dim, *hidden_sizes, action_dim],
            activation=activation,
            output_activation=output_activation,
        )
        self.log_sigma = torch.nn.Parameter(
            -0.5 * torch.ones(action_dim, dtype=torch.float), requires_grad=True
        )

    def forward(self, state: State, action: Action = None) -> (Action, LogProb):
        mu = self.action_limit * self.mu(state)
        std = torch.exp(self.log_sigma)
        dist = Normal(mu, std)

        if action is None:
            action = dist.rsample()
        logprob = dist.log_prob(action).sum(dim=-1)

        return action, logprob


class DeterministicGaussianActor(Actor):
    """Actor with a continuous action space, where actions are modeled as a
    Gaussian process.  The policy is fully *deterministic*, and the actor network
    only predicts the expected (mean) action value.  Exploration noise is typically
    added explicitly during training (as in TD3, DDPG).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.Tanh(),
        output_activation: Callable = nn.Identity(),
        action_limit: float = 1.0,
    ):
        """
        Parameters
        ----------
        state_dim: (int) Size of the state space
        action_dim: (int) Size of the action space
        hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.
            The first value should be the size of the input array.
        activation: (Callable) Activation function applied to the output of
            each layer
        output_activation: (Callable, optional) Activation function applied to the
            final output of the multi-layer perceptron.  Default: nn.Identity().
        action_limit: (float, optional) Scales the range of actions returned by
            the agent.  Default: 1.0.
        """
        super().__init__()
        self.action_limit = action_limit
        self.layers = mlp(
            [state_dim, *hidden_sizes, action_dim],
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, state: State, action: Action = None) -> (Action, None):
        return self.action_limit * self.layers(state), None


class SquashedGaussianActor(Actor):
    """Actor with a continuous action space, where actions are modeled as a
    Gaussian process.  The policy is stochastic, and the actor network simply
    predicts the mean and standard deviation of the Gaussian distribution to
    sample from.

    NOTE:
        * Default network type for SAC algorithm, for training stability reasons.
        * This differs from 'GaussianActor' only in the way that the log action
          probability is computed.  Rather than assuming a pure Gaussian
          distribution, this actor accounts for "squashing" of the distribution
          due to (typically) 'tanh' or 'relu' activation functions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        action_limit: float = 1.0,
    ):
        """
        Parameters
        ----------
        state_dim: (int) Size of the state space
        action_dim: (int) Size of the action space
        hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.
            The first value should be the size of the input array.
        action_limit: (float, optional) Scales the range of actions returned by
            the agent.  Default: 1.0.
        """
        super().__init__()
        self.action_limit = action_limit
        self.mu = mlp([state_dim, *hidden_sizes, action_dim], nn.ReLU())
        self.log_sigma = torch.nn.Parameter(
            -0.5 * torch.ones(action_dim, dtype=torch.float), requires_grad=True
        )

    def forward(self, state: State, action: Action = None) -> (Action, LogProb):
        mu = self.mu(state)
        sigma = self.log_sigma.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = Normal(mu, sigma)
        if action is None:
            action = dist.rsample()

        # Compute gaussian logprob, and then apply correction for tanh squashing.
        # NOTE: See the original SAC paper (arXiv 1801.01290), appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logprob = dist.log_prob(action).sum(dim=-1)
        logprob -= 2 * (log(2) - action - f.softplus(-2 * action)).sum(dim=-1)
        action = self.action_limit * torch.tanh(action)

        return action, logprob


# ----------------------- Standardized *CRITIC* Modules -----------------------


class Critic(nn.Module, ABC):
    """Base class for all Critic networks.  Really, this doesn't impose any
    additional abstract methods compared to 'nn.Module', but we define the return
    types for the network here as well.

    NOTE:
        * The 'forward' method should always return one Tensor: the expected
          value of the given action-state pair.
        * For discrete action spaces, we often return an array of state-action
          values -- one for each possible action.  (This is commonly known as a
          Q-Network architecture).  In these cases, we do not need the action
          as input -- since we ultimately compute the value for *every* possible
          action.  In practice, this is typically easier to train than Critic
          networks (in the traditional actor-critic sense, where the critic
          receives the chosen action as input), because actions are just sparse,
          one-hot vectors.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: State, action: Action) -> Value:
        r"""Estimates the expected value of a state-action pair."""


class GaussianCritic(Critic):
    """Critic network for continuous action spaces, where the action is modeled
    as a Gaussian process.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.Tanh(),
    ):
        """
        Parameters
        ----------
        state_dim: (int) Size of the state space
        action_dim: (int) Size of the action space
        hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.
            The first value should be the size of the input array.
        activation: (Callable) Activation function applied to the output of
            each layer
        """
        super().__init__()
        self.value = mlp(
            [state_dim + action_dim] + list(hidden_sizes) + [1], activation
        )

    def forward(self, state: State, action: Action) -> Value:
        inputs = torch.cat([state, action], dim=-1)
        return self.value(inputs).squeeze(-1)


class QNetwork(Critic):
    """Critic network for discrete action spaces (also known as a Q-Network).
    Does not need to know the chosen action, but returns an expected value for
    each possible state-action pair.

    NOTE:
        * In practice, this is typically easier to train than Critic networks (in
          the traditional actor-critic sense, where the critic receives the chosen
          action as input), because actions are just sparse, one-hot vectors.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = nn.Tanh(),
    ):
        """
        Parameters
        ----------
        state_dim: (int) Size of the state space
        action_dim: (int) Size of the action space
        hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.
            The first value should be the size of the input array.
        activation: (Callable) Activation function applied to the output of
            each layer
        """
        super().__init__()
        self.value = mlp([state_dim, *hidden_sizes, action_dim], activation)

    def forward(self, state: State, action: Action = None) -> Value:
        return self.value(state)


# ------------- Simplified functions for creating actors/critics -------------


def actor(
    env: gym.Env,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: Callable = nn.ReLU(),
    output_activation: Callable = nn.Identity(),
    action_limit: float = 1.0,
    deterministic: bool = False,
    squashed: bool = False,
) -> Actor:
    """Automatically generates an actor network for the given environment.

    Parameters
    ----------
    env: (gym.Env) Gym environment the actor will interact with
    hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.  The
        first value should be the size of the input array.
    activation: (Callable) Activation function applied to the output of each layer
    output_activation: (Callable, optional) Activation function applied to the
        final output of the multi-layer perceptron.  Default: nn.Identity()
        (i.e. no activation).  Softmax is often used for discrete action spaces.
    action_limit: (float, optional) For continuous agents, scales the range of
        actions returned by the agent.  Default: 1.0.
    deterministic: (bool, optional) If True, and the environment has a continuous
        action space, returns a 'DeterministicGaussianActor'.  Default: False.
    squashed: (bool, optional) If True, and the environment has a continuous
        action space, returns a 'SquashedGaussianActor'.  Ignored if
        'deterministic' is True.  Default: False.

    Returns
    -------
    nn.Module: Actor network
    """
    state_space = env.observation_space
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalActor(
            state_space.shape[0],
            action_space.n,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
    elif deterministic:
        return DeterministicGaussianActor(
            state_space.shape[0],
            action_space.shape[0],
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            action_limit=action_limit,
        )
    elif squashed:
        return SquashedGaussianActor(
            state_space.shape[0],
            action_space.shape[0],
            hidden_sizes=hidden_sizes,
            action_limit=action_limit,
        )
    else:
        return GaussianActor(
            state_space.shape[0],
            action_space.shape[0],
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            action_limit=action_limit,
        )


def critic(
    env: gym.Env,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: Callable = nn.Tanh(),
) -> Critic:
    """Automatically generates a critic network for the given environment.

    Parameters
    ----------
    env: (gym.Env) Gym environment the actor will interact with
    hidden_sizes: (Sequence[int], optional) Sizes of the MLP linear layers.  The
        first value should be the size of the input array.
    activation: (Callable) Activation function applied to the output of each layer

    Returns
    -------
    nn.Module: Actor network
    """
    state_space = env.observation_space
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        return GaussianCritic(
            state_space.shape[0], action_space.shape[0], hidden_sizes, activation
        )
    else:
        return QNetwork(state_space.shape[0], action_space.n, hidden_sizes, activation)
