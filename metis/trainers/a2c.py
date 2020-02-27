from typing import Iterable, Callable

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.replay import NoReplay
from metis.agents import QNetwork
from metis import base, utils


def actor_loss(
    batch,
    actor: base.Actor,
    critic: base.Critic,
    gamma: float = 0.99,
    lam: float = 0.97,
) -> (Tensor, float):
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
    advantages = utils.discount_values(deltas, dones, gamma * lam)
    advantages = (advantages - advantages.mean()) / advantages.std()

    _, logprobs = actor(states, actions)
    if logprobs.ndim > 1:
        logprobs = logprobs[range(len(actions)), actions.long()]

    return -(logprobs * advantages).mean()


def critic_loss(batch, critic: base.Critic, gamma: float = 0.99) -> Tensor:
    states, actions, _, rewards, dones, _ = batch
    returns = torch.zeros_like(rewards)
    returns[:-1] = utils.discount_values(rewards, dones, gamma)[:-1]

    if isinstance(critic, QNetwork):
        values = critic(states)[range(len(actions)), actions.long()]
        return (values - returns.unsqueeze(1)).pow(2).mean()
    else:
        return (critic(states, actions) - returns).pow(2).mean()


class A2C:
    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []
        self.avg_reward = 0.0

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.replay = None

    def update(
        self,
        actor,
        critic,
        train_critic_iters: int = 80,
        gamma: float = 0.99,
        lam: float = 0.97,
    ):
        """Performs PPO update at the end of each epoch using training samples
        that have been collected in `self.replay`.

        Parameters
        ----------
        actor
        critic
        train_critic_iters: (int) Max number of critic training steps per epoch.
        gamma: (float) Discount factor. Range: (0, 1)
        lam: (float) Hyperparameter for GAE-Lambda calaulation. Range: (0, 1)
        """
        batch = self.replay.sample()

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
        max_episode_len: int = 1000,
        gamma: float = 0.99,
        lam: float = 0.97,
        callbacks: Iterable[Callable] = (),
    ):
        """Proximal Policy Optimization (via objective clipping) with early
        stopping based on approximate KL divergence of the policy network.

        Parameters
        ----------
        actor
        critic
        replay
        actor_lr: (float) Learning rate for actor optimizer.
        critic_lr: (float) Learning rate for critic optimizer.
        train_critic_iters: (int) Max number of critic training steps per epoch.
        epochs: (int) Number of training epochs (number of policy updates)
        steps_per_epoch: (int) Number of environment steps (or turns) per epoch
        max_episode_len: (int) Max length of an environment episode (or game)
        gamma: (float) Discount factor. Range: (0, 1)
        lam:: (float) Hyperparameter for GAE-Lambda calaulation. Range: (0, 1)
        callbacks: (Iterable[Callable]) Collection of callback functions to
            execute at the end of each training epoch.
        """
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
        self.replay = replay
        if self.replay is None:
            self.replay = NoReplay(steps_per_epoch)

        for epoch in range(1, epochs + 1):
            state = self.env.reset()
            ep_reward, ep_length = 0, 0

            for t in range(1, steps_per_epoch + 1):
                with torch.no_grad():
                    action, logprob = actor(state)

                next_state, reward, done, _ = self.env.step(action)
                self.replay.append([state, action, logprob, reward, done, next_state])
                state = next_state
                ep_reward += reward
                ep_length += 1

                if done or (ep_length == max_episode_len):
                    self.ep_rewards.append(ep_reward)
                    if self.avg_reward:
                        self.avg_reward = 0.9 * self.avg_reward + 0.1 * ep_reward
                    else:
                        self.avg_reward = ep_reward
                    state = self.env.reset()
                    ep_reward, ep_length = 0, 0

            self.update(
                actor,
                critic,
                train_critic_iters=train_critic_iters,
                gamma=gamma,
                lam=lam,
            )

            print(f"\r Epoch {epoch}, Avg Reward {self.avg_reward}", end="")

            for callback in callbacks:
                callback(self)
