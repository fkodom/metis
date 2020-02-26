from typing import Iterable, Callable
from copy import deepcopy
from itertools import chain

import gym
import torch
from torch import nn, Tensor
from torch.optim import Adam

from metis import base, utils
from metis.agents import actor, critic
from metis.replay import ExperienceReplay


def actor_loss(batch, actor: base.Actor, critics: Iterable[base.Critic]) -> Tensor:
    states = batch[0]
    actions, _ = actor(states)
    return -torch.min(*[c(states, actions) for c in critics]).mean()


class TD3:
    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []
        self.avg_reward = 0.0

        self.replay = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.target_actor = None
        self.target_critics = None

    def critic_loss(
        self,
        batch,
        critics: Iterable[base.Critic],
        gamma: float = 0.99,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ) -> Tensor:
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_action, _ = self.target_actor(next_states)
            noise = torch.randn_like(next_action) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_action = next_action + noise

            next_values = torch.min(*[c(next_states, next_action) for c in self.target_critics])
            target_values = rewards + gamma * (1 - dones) * next_values

        values = [c(states, actions) for c in critics]
        return sum((value - target_values).pow(2).mean() for value in values)

    def update(
        self,
        iteration: int,
        actor: base.Actor,
        critics: Iterable[base.Critic],
        batch_size: int = 100,
        gamma: float = 0.99,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        polyak: float = 0.995,
    ):
        batch = self.replay.sample(batch_size)
        self.critic_optimizer.zero_grad()
        self.critic_loss(
            batch,
            critics,
            gamma=gamma,
            target_noise=target_noise,
            noise_clip=noise_clip,
        ).backward()
        self.critic_optimizer.step()

        if iteration % policy_delay != 0:
            return

        self.actor_optimizer.zero_grad()
        actor_loss(batch, actor, critics).backward()
        self.actor_optimizer.step()

        for p, pt in zip(actor.parameters(), self.target_actor.parameters()):
            pt.data = pt.data * polyak + (1 - polyak) * p.data
        for critic, target in zip(critics, self.target_critics):
            for p, pt in zip(critic.parameters(), target.parameters()):
                pt.data = pt.data * polyak + (1 - polyak) * p.data

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
        batch_size: int = 100,
        start_steps: int = 5000,
        update_after: int = 1000,
        update_every: int = 50,
        act_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """
        Args:
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target
                networks.
            actor_lr (float): Learning rate for policy.
            critic_lr (float): Learning rate for Q-networks.
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
            act_noise (float): Stddev for Gaussian exploration noise added to
                policy at training time. (At test time, no noise is added.)
            target_noise (float): Stddev for smoothing noise added to target
                policy.
            noise_clip (float): Limit for absolute value of target policy
                smoothing noise.
            policy_delay (int): Policy will only be updated once every
                policy_delay times for each update of the Q-networks.
        """
        self.replay = replay
        if replay is None:
            self.replay = ExperienceReplay(int(1e6))

        critic_params = chain(*[c.parameters() for c in critics])
        self.critic_optimizer = Adam(critic_params, lr=critic_lr)
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.target_actor = deepcopy(actor)
        self.target_critics = deepcopy(critics)

        total_steps = steps_per_epoch * epochs
        state, ep_reward, ep_length = self.env.reset(), 0, 0

        for step in range(total_steps):
            if step < start_steps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = actor(state)
                action += act_noise * torch.randn_like(action)

            next_state, reward, done, _ = self.env.step(action)
            done = False if ep_length == max_ep_len else done
            self.replay.append([state, action, reward, done, next_state])
            state = next_state
            ep_reward += reward
            ep_length += 1

            if step >= update_after and step % update_every == 0:
                for iter in range(update_every):
                    self.update(
                        iter,
                        actor,
                        critics,
                        batch_size=batch_size,
                        gamma=gamma,
                        target_noise=target_noise,
                        noise_clip=noise_clip,
                        policy_delay=policy_delay,
                        polyak=polyak,
                    )

            if done or (ep_length == max_ep_len):
                epoch = (step + 1) // steps_per_epoch
                print(f"\rEpoch {epoch} | Steps {step + 1} | Reward {ep_reward}", end="")
                state, ep_reward, ep_length = self.env.reset(), 0, 0

            for callback in callbacks:
                callback(self)


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    actor = actor(env, output_activation=nn.Tanh, deterministic=True)
    critics = [critic(env), critic(env)]
    TD3(env).train(actor, critics)
