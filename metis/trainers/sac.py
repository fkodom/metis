"""
metis/trainers/sac.py
---------------------
Soft actor-critic algorithm for training RL agents in both continuous and
discrete action spaces.
"""

from typing import Union, Iterable, Sequence, Callable
from copy import deepcopy
from itertools import chain

import gym
import torch
from torch import Tensor

from metis.agents import Actor, Critic, DQNCritic
from metis.replay import Replay, ExperienceReplay
from metis import utils


def actor_loss(
    batch: Sequence[Tensor or Sequence[Tensor]],
    actor: Actor,
    critics: Iterable[Critic],
    alpha: Union[float, Tensor] = 0.2,
) -> (Tensor, Tensor):
    """Computes loss for actor network.

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
        experiences for the agent being trained.
    actor: (base.Actor) Actor (policy) network to optimize.
    critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
        SAC there are *two* critics, but this method only requires that *two or
        more* critics are provided.
    alpha: (float, optional) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Default: 0.2.
    """
    states = batch[0]
    actions, logprobs = actor(states)
    if any(isinstance(c, DQNCritic) for c in critics):
        values = torch.min(*[c(states) for c in critics])
        return (logprobs.exp() * (alpha * logprobs - values)).mean()
    else:
        values = torch.min(*[c(states, actions) for c in critics])
        return (alpha * logprobs - values).mean()


class SAC:
    """Soft actor-critic algorithm for training RL agents in both continuous and
    discrete action spaces.  (arxiv:1801.01290 [cs.LG])

    SAC is very sample efficient, compared to other actor-critic algorithms like
    A3C or PPO, because it repeatedly samples from past experiences using an
    Experience Replay.  This is made possible by including *target* networks,
    which are used to bootstrap the action values for training the policy.  The
    actor network uses a *stochastic* policy, where the action uncertainty is
    parameterized by the network (not artificially added, as in DDPG or TD3).
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []

        self.replay = None
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.target_critics = None

    def critic_loss(
        self,
        batch: Sequence[Tensor or Sequence[Tensor]],
        actor: Actor,
        critics: Iterable[Critic],
        gamma: float = 0.99,
        alpha: Union[float, Tensor] = 0.2,
    ) -> Tensor:
        """Computes loss for critic networks when they are not instances of 'QNetwork'
        (i.e. the critics return a single value for the chosen action)

        Parameters
        ----------
        batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
            experiences for the agent being trained.
        actor: (base.Actor) Actor (policy) network to optimize.
        critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        alpha: (float, optional) Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  Default: 0.2.
        """
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_actions, next_logprobs = actor(next_states)
            next_values = torch.min(
                *[c(next_states, next_actions) for c in self.target_critics]
            )
            backup = next_values - alpha * next_logprobs.view(-1, 1)
            target_values = rewards + (1.0 - dones.float()) * gamma * backup

        values = [c(states, actions) for c in critics]
        return sum((value - target_values).pow(2).mean() for value in values)

    def q_network_loss(
        self,
        batch: Sequence[Tensor or Sequence[Tensor]],
        actor: Actor,
        critics: Iterable[Critic],
        gamma: float = 0.99,
        alpha: Union[float, Tensor] = 0.2,
    ) -> Tensor:
        """Computes loss for critic networks when they are instances of 'QNetwork'
        (i.e. the critics return an array of values, one for each possible discrete
        action, rather than a single value for the chosen action)

        Parameters
        ----------
        batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
            experiences for the agent being trained.
        actor: (base.Actor) Actor (policy) network to optimize.
        critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        alpha: (float, optional) Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  Default: 0.2.
        """
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_actions, next_logprobs = actor(next_states)
            next_values = torch.min(*[c(next_states) for c in self.target_critics])
            backup = (next_values - alpha * next_logprobs).mean(-1)
            target_values = rewards + (1.0 - dones.float()) * gamma * backup

        values = [c(states)[range(len(actions)), actions.long()] for c in critics]
        return sum((value - target_values).pow(2).mean() for value in values)

    def update(
        self,
        actor: Actor,
        critics: Iterable[Critic],
        batch_size: int = 128,
        gamma: float = 0.99,
        alpha: Union[float, Tensor] = 0.2,
        polyak: float = 0.995,
    ):
        """Samples from the experience replay and performs a single SAC update.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        polyak: (float, optional) Interpolation factor in polyak averaging for
            target networks.  Range: (0, 1).  Default: 0.995
        alpha: (float, optional) Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  Default: 0.2.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        """
        device = utils.get_device(actor)
        batch = self.replay.sample(batch_size, device=device)

        self.critic_optimizer.zero_grad()
        if any(isinstance(c, DQNCritic) for c in critics):
            self.q_network_loss(batch, actor, critics, alpha=alpha).backward()
        else:
            self.critic_loss(batch, actor, critics, gamma=gamma, alpha=alpha).backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss(batch, actor, critics, alpha=alpha).backward()
        self.actor_optimizer.step()

        for critic, target in zip(critics, self.target_critics):
            for p, pt in zip(critic.parameters(), target.parameters()):
                pt.data = (1 - polyak) * p.data + polyak * pt.data

    def train(
        self,
        actor: Actor,
        critics: Iterable[Critic],
        replay: Replay = None,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        gamma: float = 0.99,
        polyak: float = 0.995,
        actor_lr: float = 5e-4,
        critic_lr: float = 1e-3,
        alpha: float = 0.2,
        batch_size: int = 128,
        start_steps: int = 4000,
        update_after: int = 1000,
        update_every: int = 1,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """Soft actor-critic (SAC) training algorithm.  Supports both continuous
        and discrete action spaces.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        replay: (base.Replay, optional) Experience replay object for sampling
            previous experiences.  If not provided, defaults to 'ExperienceReplay'
            with a buffer size of 1,000,000.  Users can provide a replay object,
            which is pre-populated with experiences (for specific use cases).
        steps_per_epoch: (int, optional) Number of steps of interaction
            for the agent and the environment in each epoch.  Default: 4000.
        epochs: (int, optional) Number of training epochs.  Default:  100.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        polyak: (float, optional) Interpolation factor in polyak averaging for
            target networks.  Range: (0, 1).  Default: 0.995
        actor_lr: (float, optional) Learning rate actor optimizer.  Default: 1e-3.
        critic_lr: (float, optional) Learning rate critic optimizer.  Default: 1e-3.
        alpha: (float, optional) Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  Default: 0.2.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        start_steps: (int, optional) Number of steps for random action selection
            before running real policy (helps exploration).  Default: 1000.
        update_after: (int, optional) Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.  Default: 5000.
        update_every: (int, optional) Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.  Default: 1.
        max_ep_len: (int, optional) Maximum length of episode.  Defaults to 1000,
            but *this should be provided for each unique environment!*  This
            has an effect on how end-of-episode rewards are computed.
        callbacks: (Iterable[Callable], optional) callback functions to execute
            at the end of each training epoch.
        """
        device = utils.get_device(actor)
        self.replay = replay
        if replay is None:
            self.replay = ExperienceReplay(int(1e6))

        critic_params = chain(*[c.parameters() for c in critics])
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.target_critics = deepcopy(critics)

        state = self.env.reset()
        ep_reward, ep_length = 0, 0
        total_steps = steps_per_epoch * epochs

        for step in range(1, total_steps + 1):
            if step < start_steps:
                action = self.env.action_space.sample()
            else:
                action, _ = actor(state.to(device))

            next_state, reward, done, _ = self.env.step(action)
            done = False if ep_length == max_ep_len else done
            self.replay.append([state, action, reward, done, next_state])
            state = next_state
            ep_reward += reward
            ep_length += 1

            if step > update_after and step % update_every == 0:
                for j in range(update_every):
                    self.update(
                        actor,
                        critics,
                        batch_size=batch_size,
                        gamma=gamma,
                        alpha=alpha,
                        polyak=polyak,
                    )

            if step % steps_per_epoch == 0:
                for callback in callbacks:
                    callback(self)

            if done or (ep_length == max_ep_len):
                self.ep_rewards.append(ep_reward)
                epoch = (step + 1) // steps_per_epoch
                print(f"\rEpoch {epoch} | Step {step} | Reward {ep_reward}", end="")
                state, ep_reward, ep_length = self.env.reset(), 0, 0
