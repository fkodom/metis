from typing import Union, Iterable, Callable
from copy import deepcopy
from itertools import chain

import gym
import torch
from torch import Tensor

from metis import base, utils
from metis.replay import ExperienceReplay
from metis.agents import QNetwork


def actor_loss(
    batch,
    actor: base.Actor,
    critics: Iterable[base.Critic],
    alpha: Union[float, Tensor] = 0.2,
) -> (Tensor, Tensor):
    states = batch[0]
    actions, logprobs = actor(states)
    if any(isinstance(c, QNetwork) for c in critics):
        values = torch.min(*[c(states) for c in critics])
        return (logprobs.exp() * (alpha * logprobs - values)).mean()
    else:
        values = torch.min(*[c(states, actions) for c in critics])
        return (alpha * logprobs - values).mean()


class SAC:
    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []
        self.avg_reward = 0.0

        self.replay = None
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.target_critics = None

    def critic_loss(
        self,
        batch,
        actor: base.Actor,
        critics: Iterable[base.Critic],
        gamma: float = 0.99,
        alpha: Union[float, Tensor] = 0.2,
    ) -> Tensor:
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_actions, next_logprobs = actor(next_states)
            next_values = torch.min(*[c(next_states, next_actions) for c in self.target_critics])
            backup = next_values - alpha * next_logprobs.view(-1, 1)
            target_values = rewards + (1.0 - dones) * gamma * backup

        values = [c(states, actions) for c in critics]
        return sum((value - target_values).pow(2).mean() for value in values)

    def q_network_loss(
        self,
        batch,
        actor: base.Actor,
        critics: Iterable[base.Critic],
        gamma: float = 0.99,
        alpha: Union[float, Tensor] = 0.2,
    ) -> Tensor:
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_actions, next_logprobs = actor(next_states)
            next_values = torch.min(*[c(next_states) for c in self.target_critics])
            backup = (next_values - alpha * next_logprobs).mean(-1)
            target_values = rewards + (1.0 - dones) * gamma * backup

        values = [c(states)[range(len(actions)), actions.long()] for c in critics]
        return sum((value - target_values).pow(2).mean() for value in values)

    def update(
        self,
        actor: base.Actor,
        critics: Iterable[base.Critic],
        batch_size: int = 128,
        gamma: float = 0.99,
        alpha: Union[float, Tensor] = 0.2,
        polyak: float = 0.995,
    ):
        batch = self.replay.sample(batch_size)

        self.critic_optimizer.zero_grad()
        if any(isinstance(c, QNetwork) for c in critics):
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
        actor: base.Actor,
        critics: Iterable[base.Critic],
        replay: base.Replay = None,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        gamma: float = 0.99,
        polyak: float = 0.995,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        alpha: float = 0.2,
        batch_size: int = 128,
        start_steps: int = 4000,
        update_after: int = 1000,
        update_every: int = 1,
        max_ep_len: int = 200,
        callbacks: Iterable[Callable] = (),
    ):
        """
        Args:
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target
                networks. Range: (0, 1)
            actor_lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
        """
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

        for step in range(total_steps):
            if step < start_steps:
                action = torch.as_tensor(self.env.action_space.sample())
            else:
                with torch.no_grad():
                    action, _ = actor(state.unsqueeze(0))
                    action = torch.as_tensor(action).cpu()[0]

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

            if done or (ep_length == max_ep_len):
                epoch = (step + 1) // steps_per_epoch
                print(f"\rEpoch {epoch} | Steps {step + 1} | Reward {ep_reward}", end="")
                state, ep_reward, ep_length = self.env.reset(), 0, 0

            for callback in callbacks:
                callback(self)
