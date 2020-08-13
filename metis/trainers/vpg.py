"""
metis/trainers/vpg.py
---------------------
Vanilla Policy Gradients (VPG) algorithm for training RL agents in both
continuous and discrete action spaces.
"""

from typing import Iterable, Callable, List

import gym
from torch import Tensor
from torch.optim import Adam, Optimizer

from metis.agents import Actor
from metis.dtypes import Batch
from metis.replay import Replay, NoReplay
from metis import utils


def actor_loss(
    batch: Batch, actor: Actor, gamma: float = 0.99,
) -> Tensor:
    """Computes loss for the actor network.

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Experience sampled for training.
    actor: (base.Actor) Actor (policy) network to optimize.
    gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99

    Returns
    -------
    Tensor:  Actor loss
    """
    states, actions, rewards, dones = batch
    values = utils.discount_values(rewards, dones, gamma).to(rewards.device)
    values = (values - values.mean()) / values.std()
    _, logprobs = actor(states, actions)
    if logprobs.ndim > 1:
        logprobs = logprobs[range(len(actions)), actions.long()]

    return -(logprobs * values).mean()


class VPG:
    """Vanilla Policy Gradients algorithm for training RL agents in both
    continuous and discrete action spaces.  Paper: *Policy Gradient Methods
    for Reinforcement Learning with Function Approximation*, Sutton et al, 2000

    The original deep reinforcement learning algorithm, sometimes also known as
    REINFORCE.  This algorithm is rarely used in production these days, but it
    has *tremendous* historical and conceptual value.  VPG is very well
    theoretically supported, and its loss function is fairly simple to derive
    from first principles.  For that reason, all modern RL algorithms are
    connected on VPG in their own way (yes, including Q-networks).
    """

    def __init__(self, env: gym.Env):
        self.env = utils.torchenv(env)
        self.ep_rewards: List[float] = []
        self.avg_reward = 0.0

        self.optimizer: Optimizer = Adam([])
        self.replay: Replay = NoReplay(1)

    def update(self, actor: Actor, gamma: float = 0.99):
        """Performs PG update at the end of each epoch using training samples
        that have been collected in `self.replay`.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99.
        """
        device = utils.get_device(actor)
        batch = self.replay.sample(device=device)
        assert len(batch) == 4

        self.optimizer.zero_grad()
        actor_loss(batch, actor, gamma=gamma).backward()
        self.optimizer.step()

    def train(
        self,
        actor: Actor,
        replay: Replay = None,
        lr: float = 3e-4,
        epochs: int = 200,
        steps_per_epoch: int = 4000,
        max_ep_len: int = 1000,
        gamma: float = 0.99,
        callbacks: Iterable[Callable] = (),
    ):
        """Vanilla Policy Gradients algorithm with no added bells or whistles.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        replay: (base.Replay, optional) Experience replay object for sampling
            previous experiences.  If not provided, defaults to 'ExperienceReplay'
            with a buffer size of 1,000,000.  Users can provide a replay object,
            which is pre-populated with experiences (for specific use cases).
        steps_per_epoch: (int, optional) Number of steps of interaction
            for the agent and the environment in each epoch.  Default: 4000.
        epochs: (int, optional) Number of training epochs.  Default:  100.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        max_ep_len: (int, optional) Maximum length of episode.  Defaults to 1000,
            but *this should be provided for each unique environment!*  This
            has an effect on how end-of-episode rewards are computed.
        callbacks: (Iterable[Callable], optional) callback functions to execute
            at the end of each training epoch.
        """
        device = utils.get_device(actor)
        self.optimizer = Adam(actor.parameters(), lr=lr)
        self.replay = NoReplay(steps_per_epoch) if replay is None else replay

        for epoch in range(1, epochs + 1):
            state = self.env.reset()
            ep_reward, ep_length = 0, 0
            num_episodes = 0

            for t in range(1, steps_per_epoch + 1):
                action, _ = actor(state.to(device))

                state, reward, done, _ = self.env.step(action)
                self.replay.append([state, action, reward, done])
                ep_reward += reward
                ep_length += 1

                if done or (ep_length == max_ep_len):
                    num_episodes += 1
                    self.ep_rewards.append(ep_reward)
                    state = self.env.reset()
                    ep_reward, ep_length = 0, 0

            self.update(actor, gamma=gamma)
            avg_reward = sum(self.ep_rewards[-num_episodes:]) / num_episodes
            print(f"\rEpoch {epoch} | Avg Reward {avg_reward}", end="")

            for callback in callbacks:
                callback(self)
