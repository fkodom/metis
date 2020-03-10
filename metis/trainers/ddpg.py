"""
metis/trainers/ddpg.py
---------------------
Deep Deterministic Policy Gradients (DDPG) algorithm for training RL agents in
continuous action spaces.  Requires the actor network to have a *deterministic*
policy (i.e. no random sampling -- predicts the expected value).
"""

from typing import Iterable, Sequence, Callable
from copy import deepcopy

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.agents import Actor, Critic
from metis.replay import Replay, ExperienceReplay
from metis import utils


def actor_loss(
    batch: Sequence[Tensor or Sequence[Tensor]], actor: Actor, critic: Critic
) -> Tensor:
    """Computes loss for actor network.

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
        experiences for the agent being trained.
    actor: (base.Actor) Actor (policy) network to optimize.
    critic: (Iterable[base.Critic]) Critic networks to optimize. In standard
        SAC there are *two* critics, but this method only requires that *two or
        more* critics are provided.
    """
    states = batch[0]
    actions, _ = actor(states)
    return -critic(states, actions).mean()


class DDPG:
    """Deep Deterministic Policy Gradients (DDPG) algorithm for training RL
    agents in continuous action spaces.  Requires the actor network to have a
    *deterministic* policy (i.e. no random sampling -- predicts the expected
    value).  (arxiv:1509.02971v6 [cs.LG])

    DDPG was a landmark paper in reinforcement learning for continuous action
    spaces.  It was the first to introduce *target networks* for actor-critic
    methods, which were modeled after contemporary advances in Deep-Q Networks
    (DQNs). DDPG is very sample efficient, compared to other actor-critic
    algorithms like A3C or PPO, because it repeatedly samples from past
    experiences using an Experience Replay.  The actor network uses a
    *deterministic* policy, where the action uncertainty is artificially added
    from a Gaussian distribution during training.  This explicit randomness can
    be helpful for early exploration, but convergence is trickier than in other
    actor-critic algorithms, such as SAC or PPO.

    **NOTE:**
    DDPG is no longer the state-of-the-art for learning in continuous action
    spaces -- it has been surpassed by SAC and TD3 (it's direct successor).
    Although DDPG achieved nice sample efficiency, it is known to be highly
    sensitive to hyperparameters.  SAC and TD3 both improve upon the ideas from
    DDPG, and they are much more robust.
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []

        self.replay = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.target_actor = None
        self.target_critic = None

    def critic_loss(
        self,
        batch: Sequence[Tensor or Sequence[Tensor]],
        critic: Critic,
        gamma: float = 0.99,
    ) -> Tensor:
        """Computes loss for critic network.

        Parameters
        ----------
        batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
            experiences for the agent being trained.
        critic: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        """
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_actions, _ = self.target_actor(next_states)
            next_values = self.target_critic(next_states, next_actions)
            backup = rewards + gamma * (1 - dones.float()) * next_values

        return (critic(states, actions) - backup).pow(2).mean()

    def update(
        self,
        actor: Actor,
        critic: Critic,
        batch_size: int = 128,
        gamma: float = 0.99,
        polyak: float = 0.995,
    ):
        """Samples from the experience replay and performs a single DDPG update.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critic: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        polyak: (float, optional) Interpolation factor in polyak averaging for
            target networks.  Range: (0, 1).  Default: 0.995
        """
        device = utils.get_device(actor)
        batch = self.replay.sample(batch_size, device=device)

        self.critic_optimizer.zero_grad()
        self.critic_loss(batch, critic, gamma=gamma,).backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss(batch, actor, critic).backward()
        self.actor_optimizer.step()

        for p, pt in zip(actor.parameters(), self.target_actor.parameters()):
            pt.data = pt.data * polyak + (1 - polyak) * p.data
        for p, pt in zip(critic.parameters(), self.target_critic.parameters()):
            pt.data = pt.data * polyak + (1 - polyak) * p.data

    def train(
        self,
        actor: Actor,
        critic: Critic,
        replay: Replay = None,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        gamma: float = 0.99,
        polyak: float = 0.995,
        actor_lr: float = 5e-4,
        critic_lr: float = 1e-3,
        batch_size: int = 128,
        start_steps: int = 5000,
        update_after: int = 1000,
        update_every: int = 50,
        act_noise: float = 0.1,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """Deep Deterministic Policy Gradients (DDPG) training algorithm.
        Supports only *deterministic* policies in *continuous* action spaces.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critic: (Iterable[base.Critic]) Critic networks to optimize. In standard
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
        act_noise: (float, optional) Stddev for Gaussian exploration noise added
            to policy at training time.  Default: 0.1.
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

        self.critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)

        total_steps = steps_per_epoch * epochs
        state, ep_reward, ep_length = self.env.reset(), 0, 0

        for step in range(1, total_steps + 1):
            if step < start_steps:
                action = self.env.action_space.sample()
            else:
                action, _ = actor(state.to(device))
                action += act_noise * torch.randn_like(action)

            next_state, reward, done, _ = self.env.step(action)
            done = False if ep_length == max_ep_len else done
            self.replay.append([state, action, reward, done, next_state])
            state = next_state
            ep_reward += reward
            ep_length += 1

            if step >= update_after and step % update_every == 0:
                for _ in range(update_every):
                    self.update(
                        actor,
                        critic,
                        batch_size=batch_size,
                        gamma=gamma,
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
