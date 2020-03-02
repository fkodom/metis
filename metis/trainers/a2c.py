"""
metis/trainers/a2c.py
---------------------
Advantage Actor-Critic (A2C) algorithm for training RL agents in both
continuous and discrete action spaces.
"""

from typing import Iterable, Sequence, Callable

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.replay import NoReplay
from metis.agents import QNetwork
from metis import base, utils


def actor_loss(
    batch: Sequence[Tensor or Sequence[Tensor]],
    actor: base.Actor,
    critic: base.Critic,
    gamma: float = 0.99,
    lam: float = 0.97,
) -> Tensor:
    """Computes loss for the actor network.

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Experience sampled for training.
    actor: (base.Actor) Actor (policy) network to optimize.
    critic: (base.Critic) Critic network to optimize.
    gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
    lam: (float, optional) Hyperparameter for GAE-Lambda calaulation.
        Range: (0, 1).  Default: 0.97

    Returns
    -------
    (Tensor, float):  Actor loss, KL divergence
    """
    states, actions, old_logprobs, rewards, dones, next_states = batch
    with torch.no_grad():
        if isinstance(critic, QNetwork):
            values = critic(states)[range(len(actions)), actions.long()]
            next_act = actor(next_states)[0]
            next_values = critic(next_states)[range(len(next_act)), next_act.long()]
        else:
            values = critic(states, actions)
            next_act = actor(next_states)[0]
            next_values = critic(next_states, next_act)

    # GAE-Lambda advantages
    deltas = rewards + gamma * next_values - values
    deltas = torch.where(dones > 1e-6, rewards, deltas)
    advantages = utils.discount_values(deltas, dones, gamma * lam).to(deltas.device)
    advantages = (advantages - advantages.mean()) / advantages.std()

    _, logprobs = actor(states, actions)
    if logprobs.ndim > 1:
        logprobs = logprobs[range(len(actions)), actions.long()]

    return -(logprobs * advantages).mean()


def critic_loss(
    batch: Sequence[Tensor or Sequence[Tensor]],
    critic: base.Critic,
    gamma: float = 0.99,
) -> Tensor:
    """Computes loss for critic networks.

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Experience sampled for training.
    critic: (base.Critic) Critic network to optimize.
    gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99

    Returns
    -------
    Tensor:  Critic loss
    """
    states, actions, _, rewards, dones, _ = batch
    returns = torch.zeros_like(rewards)
    returns[:-1] = utils.discount_values(rewards, dones, gamma)[:-1].to(returns.device)

    if isinstance(critic, QNetwork):
        values = critic(states)[range(len(actions)), actions.long()]
        return (values - returns.unsqueeze(1)).pow(2).mean()
    else:
        return (critic(states, actions) - returns).pow(2).mean()


class A2C:
    """Advantage Actor-Critic (A2C) algorithm for training RL agents in
    both continuous and discrete action spaces.  (arxiv:1602.01783 [cs.LG])

    TODO: Add algorithm summary/description
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.replay = None

    def update(
        self,
        actor: base.Actor,
        critic: base.Critic,
        train_critic_iters: int = 80,
        gamma: float = 0.99,
        lam: float = 0.97,
    ):
        """Performs A2C update at the end of each epoch using training samples
        that have been collected in `self.replay`.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critic: (base.Critic) Critic network to optimize.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        train_critic_iters: (int, optional) Max number of critic training steps
            per epoch.  Default: 80.
        lam: (float, optional) Hyperparameter for GAE-Lambda calaulation.
            Range: (0, 1).  Default: 0.97
        """
        device = utils.get_device(actor)
        batch = self.replay.sample(device=device)

        self.actor_optimizer.zero_grad()
        actor_loss(batch, actor, critic, gamma=gamma, lam=lam).backward()
        self.actor_optimizer.step()

        for i in range(train_critic_iters):
            self.critic_optimizer.zero_grad()
            critic_loss(batch, critic, gamma=gamma).backward()
            self.critic_optimizer.step()

    def train(
        self,
        actor: base.Actor,
        critic: base.Critic,
        replay: base.Replay = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        train_critic_iters: int = 10,
        epochs: int = 200,
        steps_per_epoch: int = 4000,
        max_ep_len: int = 1000,
        gamma: float = 0.99,
        lam: float = 0.97,
        callbacks: Iterable[Callable] = (),
    ):
        """Advantage Actor-Critic (A2C) algorithm for training RL agents in both
        continuous and discrete action spaces.

        **NOTE:** Synchronous A2C was chosen over the asynchronous version (A3C)
        due to its simplicity.  It's also questionable that A3C performs better
        than A2C in the first place (in terms of the resulting trained policy,
        not training speed in Python).  Other people/organizations have also
        pointed this out, including OpenAI.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critic: (base.Critic) Critic network to optimize.
        replay: (base.Replay, optional) Experience replay object for sampling
            previous experiences.  If not provided, defaults to 'ExperienceReplay'
            with a buffer size of 1,000,000.  Users can provide a replay object,
            which is pre-populated with experiences (for specific use cases).
        steps_per_epoch: (int, optional) Number of steps of interaction
            for the agent and the environment in each epoch.  Default: 4000.
        epochs: (int, optional) Number of training epochs.  Default:  100.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        actor_lr: (float, optional) Learning rate actor optimizer.  Default: 1e-3.
        critic_lr: (float, optional) Learning rate critic optimizer.  Default: 1e-3.
        train_critic_iters: (int, optional) Max number of critic training steps
            per epoch.  Default: 10.
        lam: (float, optional) Hyperparameter for GAE-Lambda calaulation.
            Range: (0, 1).  Default: 0.97.
        max_ep_len: (int, optional) Maximum length of episode.  Defaults to 1000,
            but *this should be provided for each unique environment!*  This
            has an effect on how end-of-episode rewards are computed.
        callbacks: (Iterable[Callable], optional) callback functions to execute
            at the end of each training epoch.
        """
        device = utils.get_device(actor)
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
        self.replay = replay
        if self.replay is None:
            self.replay = NoReplay(steps_per_epoch)

        for epoch in range(1, epochs + 1):
            state = self.env.reset()
            ep_reward, ep_length = 0, 0
            num_episodes = 0

            for t in range(1, steps_per_epoch + 1):
                action, logprob = actor(state.to(device))

                next_state, reward, done, _ = self.env.step(action)
                self.replay.append([state, action, logprob, reward, done, next_state])
                state = next_state
                ep_reward += reward
                ep_length += 1

                if done or (ep_length == max_ep_len):
                    num_episodes += 1
                    self.ep_rewards.append(ep_reward)
                    state = self.env.reset()
                    ep_reward, ep_length = 0, 0

            self.update(
                actor,
                critic,
                train_critic_iters=train_critic_iters,
                gamma=gamma,
                lam=lam,
            )

            avg_reward = sum(self.ep_rewards[-num_episodes:]) / num_episodes
            print(f"\rEpoch {epoch} | Avg Reward {avg_reward}", end="")

            for callback in callbacks:
                callback(self)
