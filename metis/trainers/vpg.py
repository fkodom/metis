from typing import Iterable, Callable

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.replay import NoReplay
from metis import base, utils


def actor_loss(
    batch,
    actor: base.Actor,
    gamma: float = 0.99,
) -> Tensor:
    states, actions, rewards, dones = batch
    values = utils.discount_values(rewards, dones, gamma)
    values = (values - values.mean()) / values.std()
    _, logprobs = actor(states, actions)

    return -(logprobs * values).mean()


class VPG:
    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []
        self.avg_reward = 0.0

        self.optimizer = None
        self.replay = None

    def update(
        self,
        actor,
        gamma: float = 0.99,
    ):
        """Performs PPO update at the end of each epoch using training samples
        that have been collected in `self.replay`.

        Parameters
        ----------
        actor
        gamma: (float) Discount factor. Range: (0, 1)
        """
        batch = self.replay.sample()
        self.optimizer.zero_grad()
        actor_loss(batch, actor, gamma=gamma).backward()
        self.optimizer.step()

    def train(
        self,
        actor: base.Actor,
        replay: base.Replay = None,
        lr: float = 3e-4,
        epochs: int = 200,
        steps_per_epoch: int = 4000,
        max_episode_len: int = 1000,
        gamma: float = 0.99,
        callbacks: Iterable[Callable] = (),
    ):
        """Proximal Policy Optimization (via objective clipping) with early
        stopping based on approximate KL divergence of the policy network.

        Parameters
        ----------
        actor
        replay
        lr: (float) Learning rate for actor optimizer.
        epochs: (int) Number of training epochs (number of policy updates)
        steps_per_epoch: (int) Number of environment steps (or turns) per epoch
        max_episode_len: (int) Max length of an environment episode (or game)
        gamma: (float) Discount factor. Range: (0, 1)
        callbacks: (Iterable[Callable]) Collection of callback functions to
            execute at the end of each training epoch.
        """
        self.optimizer = Adam(actor.parameters(), lr=lr)
        self.replay = replay
        if self.replay is None:
            self.replay = NoReplay(steps_per_epoch)

        for epoch in range(1, epochs + 1):
            state = self.env.reset()
            ep_reward, ep_length = 0, 0

            for t in range(1, steps_per_epoch + 1):
                with torch.no_grad():
                    action, _ = actor(state)

                state, reward, done, _ = self.env.step(action)
                self.replay.append([state, action, reward, done])
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

            self.update(actor, gamma=gamma)
            print(f"\r Epoch {epoch}, Avg Reward {self.avg_reward}", end="")
            for callback in callbacks:
                callback(self)
