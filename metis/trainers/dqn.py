"""
metis/trainers/dqn.py
---------------------
Deep Q-Network (DQN) algorithm for training RL agents in discrete action spaces.
"""

from typing import Iterable, Sequence, Callable
from copy import deepcopy
from math import exp

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis import utils, agents
from metis.replay import Replay, PER


class DQN:
    """Deep Q-Network (DQN) algorithm for training RL agents in discrete action
    spaces.  Only uses a policy/actor network (i.e. no critic), which returns an
    array of state-action values (one for each possible action).  This type of
    policy is commonly known as a Q-Network.  (arxiv:1312.5602v1 [cs.LG])

    DQN marked the start of "modern" reinforcement learning.  In the landmark
    paper from DeepMind, they demonstrated the ability to play Atari games at a
    high level using DQN.  Their trained agents matched or exceeded human-level
    performance in several Atari environments.  Unlike policy gradients methods,
    DQN *approximates* the optimal policy by estimating the value of each
    state-action pair.  It also achieves very good sample efficiency by using an
    experience replay buffer.  Several improvements have been made upon the
    original DQN paper (e.g. DDQN, Dueling DQN, etc.), but this model will
    always hold an important place in history and performannce baselines.
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []

        self.replay = None
        self.optimizer = None
        self.target_dqn = None

    def dqn_loss(
        self,
        batch: Sequence[Tensor or Sequence[Tensor]],
        dqn: agents.DQN,
        gamma: float = 0.99,
    ) -> Tensor:
        """Computes loss for DQN network.

        Parameters
        ----------
        batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
            experiences for the agent being trained.
        dqn: (agents.DQN) DQN network used to select training actions, which will
            have its parameters updated directly.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        """
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            next_actions, next_values = self.target_dqn(next_states)
            next_values = next_values[range(len(next_actions)), next_actions.long()]
            backup = rewards + gamma * (1 - dones.float()) * next_values

        _, values = dqn(states)
        values = values[range(len(actions)), actions.long()]
        td_errors = (values - backup)
        self.replay.update(td_errors.abs())

        return td_errors.pow(2).mean()

    def update(
        self,
        iteration: int,
        dqn: agents.DQN,
        batch_size: int = 128,
        gamma: float = 0.99,
        target_update: int = 10,
    ):
        """Samples from the experience replay and performs a single update.

        Parameters
        ----------
        iteration: (int) Number of update iterations that have been performed
            during this update step.  Used for updating target DQN network.
        dqn: (agents.DQN) DQN network used to select training actions, which will
            have its parameters updated directly.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        target_update: (int, optional) Frequency at which the target DQN network
            should be updated.  Should generally be larger than 1.  Default: 10.
        """
        device = utils.get_device(dqn)
        batch = self.replay.sample(batch_size, device=device)

        self.optimizer.zero_grad()
        self.dqn_loss(batch, dqn, gamma=gamma).backward()
        self.optimizer.step()

        if iteration % target_update == 0:
            self.target_dqn.load_state_dict(dqn.state_dict())

    def train(
        self,
        dqn: agents.DQN,
        replay: Replay = None,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 128,
        target_update: int = 10,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """Deep Q-Network (DQN) training algorithm for RL agents in discrete
        action spaces.

        Parameters
        ----------
        dqn: (agents.DQN) DQN network used to select training actions, which will
            have its parameters updated directly.
        replay: (base.Replay, optional) Experience replay object for sampling
            previous experiences.  If not provided, defaults to 'ExperienceReplay'
            with a buffer size of 1,000,000.  Users can provide a replay object,
            which is pre-populated with experiences (for specific use cases).
        steps_per_epoch: (int, optional) Number of steps of interaction
            for the agent and the environment in each epoch.  Default: 4000.
        epochs: (int, optional) Number of training epochs.  Default:  100.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        lr: (float, optional) Learning rate critic optimizer.  Default: 1e-3.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        target_update: (int, optional) Frequency at which the target DQN network
            should be updated.  Should generally be larger than 1.  Default: 10.
        max_ep_len: (int, optional) Maximum length of episode.  Defaults to 1000,
            but *this should be provided for each unique environment!*  This
            has an effect on how end-of-episode rewards are computed.
        callbacks: (Iterable[Callable], optional) callback functions to execute
            at the end of each training epoch.
        """
        device = utils.get_device(dqn)
        self.replay = replay
        if replay is None:
            self.replay = PER(int(1e6))

        self.optimizer = Adam(dqn.parameters(), lr=lr)
        self.target_dqn = deepcopy(dqn)

        total_steps = steps_per_epoch * epochs
        state, ep_reward, ep_length = self.env.reset(), 0, 0

        for step in range(1, total_steps + 1):
            eps = 0.05 + (0.9 - 0.05) * exp(-step / 1000)
            if torch.rand(1).item() < eps:
                action = self.env.action_space.sample()
            else:
                action, _ = dqn(state.to(device))

            next_state, reward, done, _ = self.env.step(action)
            done = False if ep_length == max_ep_len else done
            self.replay.append([state, action, reward, done, next_state])
            state = next_state
            ep_reward += reward
            ep_length += 1

            if len(self.replay) >= batch_size:
                self.update(
                    step,
                    dqn,
                    batch_size=batch_size,
                    gamma=gamma,
                    target_update=target_update,
                )

            if step % steps_per_epoch == 0:
                for callback in callbacks:
                    callback(self)

            if done or (ep_length == max_ep_len):
                self.ep_rewards.append(ep_reward)
                epoch = (step + 1) // steps_per_epoch
                print(f"\rEpoch {epoch} | Step {step} | Reward {ep_reward}", end="")
                state, ep_reward, ep_length = self.env.reset(), 0, 0
