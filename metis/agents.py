from abc import ABC, abstractmethod
from typing import Sequence
from math import log

import gym
import torch
from torch import Tensor, nn
import torch.nn.functional as f
from torch.distributions import Normal, Categorical

LOG_STD_MIN = -20
LOG_STD_MAX = 2


State = Tensor or Sequence[Tensor]
Action = Tensor or float or int
LogProb = Tensor or Sequence[Tensor]
Value = Tensor


# ----------------------- Standardized *ACTOR* Modules -----------------------


class Actor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: State) -> (Action, LogProb):
        r"""Computes action probabilities for the given environment state."""


class DeterministicActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        output_activation=nn.Identity,
        action_limit=1.0,
    ):
        super().__init__()
        self.action_limit = action_limit
        self.layers = mlp(
            [obs_dim, *hidden_sizes, act_dim],
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, state, action=None) -> (Tensor, None):
        return self.action_limit * self.layers(state), None


class CategoricalActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ):
        super().__init__()
        self.logits = mlp(
            [obs_dim, *hidden_sizes, act_dim],
            activation=activation,
        )

    def forward(self, obs, action=None) -> (Tensor, Tensor):
        probs = torch.softmax(self.logits(obs), dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()

        return action, probs.log()


class GaussianActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        output_activation=nn.Identity,
        action_limit=1.0,
    ):
        super().__init__()
        self.action_limit = action_limit
        self.mu = mlp(
            [obs_dim, *hidden_sizes, act_dim],
            activation=activation,
            output_activation=output_activation,
        )
        self.log_sigma = torch.nn.Parameter(
            -0.5 * torch.ones(act_dim, dtype=torch.float), requires_grad=True
        )

    def forward(self, state, action=None) -> (Tensor, Tensor):
        mu = self.action_limit * self.mu(state)
        std = torch.exp(self.log_sigma)
        dist = Normal(mu, std)

        if action is None:
            action = dist.rsample()
        logprob = dist.log_prob(action).sum(dim=-1)

        return action, logprob


class SquashedGaussianActor(Actor):
    def __init__(
        self, obs_dim, act_dim, hidden_sizes=(64, 64), action_limit: float = 1.0
    ):
        super().__init__()
        self.action_limit = action_limit
        self.mu = mlp([obs_dim, *hidden_sizes, act_dim], nn.ReLU)
        self.log_sigma = torch.nn.Parameter(
            -0.5 * torch.ones(act_dim, dtype=torch.float), requires_grad=True
        )

    def forward(self, state, action=None) -> (Tensor, Tensor):
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
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: State, action: Action) -> Value:
        r"""Pushes the environment state through the network, and returns action
        probabilities and state value.
        """


class GaussianCritic(Critic):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.value = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act) -> Tensor:
        inputs = torch.cat([obs, act], dim=-1)
        return self.value(inputs).squeeze(-1)


class CategoricalCritic(Critic):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.value = mlp([obs_dim, *hidden_sizes, act_dim], activation)

    def forward(self, obs, action=None) -> Tensor:
        return self.value(obs)


class QNetwork(Critic):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.value = mlp([obs_dim, *hidden_sizes, act_dim], activation)

    def forward(self, obs, action=None) -> Tensor:
        return self.value(obs)


# ------------- Simplified functions for creating actors/critics -------------


def actor(
    env: gym.Env,
    hidden_sizes=(64, 64),
    activation=nn.ReLU,
    output_activation=nn.Identity,
    action_limit=1.0,
    deterministic=False,
    squashed=False,
) -> Actor:
    obs_space = env.observation_space
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Discrete):
        return CategoricalActor(
            obs_space.shape[0],
            act_space.n,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
    elif deterministic:
        return DeterministicActor(
            obs_space.shape[0],
            act_space.shape[0],
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            action_limit=action_limit,
        )
    elif squashed:
        return SquashedGaussianActor(
            obs_space.shape[0],
            act_space.shape[0],
            hidden_sizes=hidden_sizes,
            action_limit=action_limit,
        )
    else:
        return GaussianActor(
            obs_space.shape[0],
            act_space.shape[0],
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            action_limit=action_limit,
        )


def critic(env: gym.Env, hidden_sizes=(64, 64), activation=nn.Tanh) -> Critic:
    obs_space = env.observation_space
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Box):
        return GaussianCritic(obs_space.shape[0], act_space.shape[0], hidden_sizes, activation)
    else:
        return QNetwork(obs_space.shape[0], act_space.n, hidden_sizes, activation)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)