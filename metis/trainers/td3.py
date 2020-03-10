"""
metis/trainers/td3.py
---------------------
Twin-Delayed Deep Deterministic (TD3) Policy Gradients algorithm for training RL
agents in continuous action spaces.  Requires the actor network to have a
*deterministic* policy (i.e. no random sampling -- predicts the expected value).
"""

from typing import Iterable, Sequence, Callable
from copy import deepcopy
from itertools import chain

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.agents import Actor, Critic
from metis.replay import Replay, ExperienceReplay
from metis import utils


def actor_loss(
    batch: Sequence[Tensor or Sequence[Tensor]], actor: Actor, critics: Iterable[Critic]
) -> Tensor:
    """Computes loss for actor network.

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
        experiences for the agent being trained.
    actor: (base.Actor) Actor (policy) network to optimize.
    critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
        SAC there are *two* critics, but this method only requires that *two or
        more* critics are provided.
    """
    states = batch[0]
    actions, _ = actor(states)
    return -torch.min(*[c(states, actions) for c in critics]).mean()


class TD3:
    """Twin-Delayed Deep Deterministic (TD3) Policy Gradients algorithm for
    training RL agents in continuous action spaces.  Requires the actor network
    to have a *deterministic* policy (i.e. no random sampling -- predicts the
    expected value).  (arxiv:1802.09477 [cs.AI])

    TD3 is very sample efficient, compared to other actor-critic algorithms like
    A3C or PPO, because it repeatedly samples from past experiences using an
    Experience Replay.  This is made possible by including *target* networks,
    which are used to bootstrap the action values for training the policy.  The
    actor network uses a *deterministic* policy, where the action uncertainty is
    artificially added from a Gaussian distribution during training.  This
    explicit randomness can be helpful for early exploration, but convergence
    is trickier than in other actor-critic algorithms, such as SAC or PPO.
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []

        self.replay = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.target_actor = None
        self.target_critics = None

    def critic_loss(
        self,
        batch: Sequence[Tensor or Sequence[Tensor]],
        critics: Iterable[Critic],
        gamma: float = 0.99,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ) -> Tensor:
        """Computes loss for critic networks.

        Parameters
        ----------
        batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
            experiences for the agent being trained.
        critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        target_noise: (float, optional) Stddev for smoothing noise added to
            target policy.  Default: 0.2.
        noise_clip: (float, optional) Max absolute value of target policy
            smoothing noise.  Default: 0.5.
        """
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_action, _ = self.target_actor(next_states)
            noise = torch.randn_like(next_action) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_action = next_action + noise

            next_values = torch.min(
                *[c(next_states, next_action) for c in self.target_critics]
            )
            target_values = rewards + gamma * (1 - dones.float()) * next_values

        values = [c(states, actions) for c in critics]
        return sum((value - target_values).pow(2).mean() for value in values)

    def update(
        self,
        iteration: int,
        actor: Actor,
        critics: Iterable[Critic],
        batch_size: int = 128,
        gamma: float = 0.99,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        polyak: float = 0.995,
    ):
        """Samples from the experience replay and performs a single TD3 update.

        Parameters
        ----------
        iteration: (int) Number of update iterations that have been performed
            during this update step.  Used for monitoring policy update delays.
        actor: (base.Actor) Actor (policy) network to optimize.
        critics: (Iterable[base.Critic]) Critic networks to optimize. In standard
            SAC there are *two* critics, but this method only requires that *two or
            more* critics are provided.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        target_noise: (float, optional) Stddev for smoothing noise added to
            target policy.  Default: 0.2.
        noise_clip: (float, optional) Max absolute value of target policy
            smoothing noise.  Default: 0.5.
        policy_delay: (int, optional) Policy will only be updated once every
            policy_delay times for each update of the Q-networks.  Default: 2.
        polyak: (float, optional) Interpolation factor in polyak averaging for
            target networks.  Range: (0, 1).  Default: 0.995
        """
        device = utils.get_device(actor)
        batch = self.replay.sample(batch_size, device=device)

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
        actor: Actor,
        critics: Iterable[Critic],
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
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """Twin-Delayed Deep Deterministic (TD3) Policy Gradients training
        algorithm.  Supports only *deterministic* policies in *continuous*
        action spaces.

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
        target_noise: (float, optional) Stddev for smoothing noise added to
            target policy.  Default: 0.2.
        noise_clip: (float, optional) Max absolute value of target policy
            smoothing noise.  Default: 0.5.
        policy_delay: (int, optional) Policy will only be updated once every
            policy_delay times for each update of the Q-networks.  Default: 2.
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
        self.critic_optimizer = Adam(critic_params, lr=critic_lr)
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.target_actor = deepcopy(actor)
        self.target_critics = deepcopy(critics)

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

            if step % steps_per_epoch == 0:
                for callback in callbacks:
                    callback(self)

            if done or (ep_length == max_ep_len):
                self.ep_rewards.append(ep_reward)
                epoch = (step + 1) // steps_per_epoch
                print(f"\rEpoch {epoch} | Step {step} | Reward {ep_reward}", end="")
                state, ep_reward, ep_length = self.env.reset(), 0, 0
