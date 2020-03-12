"""
metis/trainers/ddqn.py
---------------------
Double Deep Q-Network (DQN) algorithm for training RL agents in discrete
action spaces.
"""

from typing import Iterable, Sequence, Callable
from math import exp

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.agents import DQN
from metis.replay import Replay, ExperienceReplay
from metis import utils


class DDQN:
    """Double Deep Q-Network (DDQN) algorithm for training RL agents in discrete
    action spaces.  Simultaneously trains two policy/actor networks (i.e. no
    critic), which return an array of state-action values (one for each possible
    action).  This type of policy is commonly known as a Q-Network.
    (arxiv:1509.06461 [cs.LG])

    DDQN presented the first significant improvement on the seminal DQN paper.
    DQN often overestimates the value of state-action pairs, which DDQN provides
    a solution for.  DDQN decouples the value estimation of DQN by simultaneously
    training two networks.  As a result, training is much more stable (although
    sometimes a bit slower starting for simple environments).  DDQN also achieves
    very good sample efficiency, like its DQN predecessor.
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards = []

        self.replay = None
        self.optimizers = []

    def dqn_loss(
        self,
        batch: Sequence[Tensor or Sequence[Tensor]],
        dqn: DQN,
        double_dqn: DQN,
        gamma: float = 0.99,
    ) -> Tensor:
        """Computes loss for DQN network.

        Parameters
        ----------
        batch: (Sequence[Tensor or Sequence[Tensor]]) Sampled batch of past
            experiences for the agent being trained.
        dqn: (agents.DQN) DQN network used to select training actions, which will
            have its parameters updated directly.
        double_dqn: (agents.DQN) DQN network used to evaluate action values, which does
            not have its parameters updated (in this update iteration).
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        """
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            _, next_values = double_dqn(next_states)
            next_actions, _ = dqn(next_states)
            next_values = next_values[range(len(next_actions)), next_actions.long()]
            backup = rewards + gamma * (1 - dones.float()) * next_values

        _, values = dqn(states)
        values = values[range(len(actions)), actions.long()]

        return (values - backup).pow(2).mean()

    def update(
        self,
        iteration: int,
        dqns: Sequence[DQN],
        batch_size: int = 128,
        gamma: float = 0.99,
    ):
        """Samples from the experience replay and performs a single DDPG update.

        Parameters
        ----------
        iteration: (int) Number of update iterations that have been performed
            during this update step.  Used for updating target DQN network.
        dqns: (Sequence[DQN]) List or tuple of all DQN networks used for training.
            Must contain a *minimum of two* DQN networks (as in the original DDQN
            paper).  In general, this function will accept more than two.  In
            that case, network (i + 1) acts as the Double DQN for network (i)
            during training -- the Double DQN relationship becomes circular.
        batch_size: (int, optional) Minibatch size for SGD.  Default: 128.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        """
        device = utils.get_device(dqns[0])
        batch = self.replay.sample(batch_size, device=device)

        dqn = dqns[iteration % len(dqns)]
        optimizer = self.optimizers[iteration % len(dqns)]
        double_dqn = dqns[(iteration + 1) % len(dqns)]

        optimizer.zero_grad()
        self.dqn_loss(batch, dqn, double_dqn, gamma=gamma).backward()
        optimizer.step()

    def train(
        self,
        dqns: Sequence[DQN],
        replay: Replay = None,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """
        Parameters
        ----------
        dqns: (Sequence[DQN]) List or tuple of all DQN networks used for training.
            Must contain a *minimum of two* DQN networks (as in the original DDQN
            paper).  In general, this function will accept more than two.  In
            that case, network (i + 1) acts as the Double DQN for network (i)
            during training -- the Double DQN relationship becomes circular.
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
        max_ep_len: (int, optional) Maximum length of episode.  Defaults to 1000,
            but *this should be provided for each unique environment!*  This
            has an effect on how end-of-episode rewards are computed.
        callbacks: (Iterable[Callable], optional) callback functions to execute
            at the end of each training epoch.
        """
        device = utils.get_device(dqns[0])
        self.replay = replay
        if replay is None:
            self.replay = ExperienceReplay(int(1e6))
        self.optimizers = [Adam(dqn.parameters(), lr=lr) for dqn in dqns]

        total_steps = steps_per_epoch * epochs
        state, ep_reward, ep_length = self.env.reset(), 0, 0

        for step in range(1, total_steps + 1):
            eps = 0.05 + (0.9 - 0.05) * exp(-step / 1000)
            if torch.rand(1).item() < eps:
                action = self.env.action_space.sample()
            else:
                dqn_idx = step % len(dqns)
                action, _ = dqns[dqn_idx](state.to(device))

            next_state, reward, done, _ = self.env.step(action)
            done = False if ep_length == max_ep_len else done
            self.replay.append([state, action, reward, done, next_state])
            state = next_state
            ep_reward += reward
            ep_length += 1

            if len(self.replay) >= batch_size:
                self.update(
                    step, dqns, batch_size=batch_size, gamma=gamma,
                )

            if step % steps_per_epoch == 0:
                for callback in callbacks:
                    callback(self)

            if done or (ep_length == max_ep_len):
                self.ep_rewards.append(ep_reward)
                epoch = (step + 1) // steps_per_epoch
                print(f"\rEpoch {epoch} | Step {step} | Reward {ep_reward}", end="")
                state, ep_reward, ep_length = self.env.reset(), 0, 0
