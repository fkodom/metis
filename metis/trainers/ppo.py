"""
metis/trainers/ppo.py
---------------------
Proximal Policy Optimization (PPO) algorithm for training RL agents in both
continuous and discrete action spaces.
"""

from typing import Iterable, Sequence, Callable

import gym
import torch
from torch import Tensor
from torch.optim import Adam

from metis.replay import NoReplay
from metis import base, utils
from metis.agents import QNetwork


def actor_loss(
    batch,
    actor: base.Actor,
    critic: base.Critic,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.97,
) -> (Tensor, float):
    """Computes loss for the actor network, as well as the approximate
    KL-divergence (used for early stopping of each training update).

    Parameters
    ----------
    batch: (Sequence[Tensor or Sequence[Tensor]]) Experience sampled for training.
    actor: (base.Actor) Actor (policy) network to optimize.
    critic: (base.Critic) Critic network to optimize.
    gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
    clip_ratio: (float, optional) Hyperparameter for clipping in the policy
        objective.  Scales how much the policy is allowed change per
        training update.  Default: 0.2.
    lam: (float, optional) Hyperparameter for GAE-Lambda calaulation.
        Range: (0, 1).  Default: 0.97

    Returns
    -------
    (Tensor, float):  Actor loss, KL divergence
    """
    states, actions, old_logprobs, rewards, dones, next_states = batch
    with torch.no_grad():
        if isinstance(critic, QNetwork):
            vals = critic(states)[range(len(actions)), actions.long()]
            next_act = actor(next_states)[0]
            next_vals = critic(next_states)[range(len(next_act)), next_act.long()]
        else:
            vals = critic(states, actions)
            next_act = actor(next_states)[0]
            next_vals = critic(next_states, next_act)

    # GAE-Lambda advantages
    deltas = rewards + gamma * next_vals - vals
    deltas = torch.where(dones > 1e-6, rewards, deltas)
    adv = utils.discount_values(deltas, dones, gamma * lam)
    adv = (adv - adv.mean()) / adv.std()

    _, logp = actor(states, actions)
    ratio = torch.exp(logp - old_logprobs)
    if ratio.ndim > 1:
        ratio = ratio[range(len(actions)), actions.long()]
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
    approx_kl = (old_logprobs - logp).mean().item()

    return loss_pi, approx_kl


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
    returns[:-1] = utils.discount_values(rewards, dones, gamma)[:-1]

    if isinstance(critic, QNetwork):
        values = critic(states)[range(len(actions)), actions.long()]
        return (values - returns.unsqueeze(1)).pow(2).mean()
    else:
        return (critic(states, actions) - returns).pow(2).mean()


class PPO:
    """Proximal Policy Optimization (PPO) algorithm for training RL agents in
    both continuous and discrete action spaces.  (arxiv:1707.06347 [cs.LG])

    PPO is known for being extremely stable during training.  It has been used
    for many famous experiments, such as
    [OpenAI Five](https://openai.com/blog/openai-five/) for that reason.  PPO
    is an on-policy algorithm, which makes it less sample efficient than other
    actor-critic models like SAC, DDPG, or TD3.  However, it is usually much
    more efficient than A2C or VPG, since each training epoch continues to update
    until a target KL divergence has been reached.  This means that samples are
    almost always used more than once before being thrown away.  The actor
    network uses a *stochastic* policy, where the action uncertainty is
    parameterized by the network (not artificially added, as in DDPG or TD3).
    """

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
        train_actor_iters: int = 80,
        train_critic_iters: int = 80,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.97,
        target_kl: float = 0.01,
    ):
        """Performs PPO update at the end of each epoch using training samples
        that have been collected in `self.replay`.

        Parameters
        ----------
        actor: (base.Actor) Actor (policy) network to optimize.
        critic: (base.Critic) Critic network to optimize.
        gamma: (float, optional) Discount factor.  Range: (0, 1).  Default: 0.99
        train_actor_iters: (int, optional) Max number of actor training steps
            per epoch.  Default: 80.
        train_critic_iters: (int, optional) Max number of critic training steps
            per epoch.  Default: 80.
        clip_ratio: (float, optional) Hyperparameter for clipping in the policy
            objective.  Scales how much the policy is allowed change per
            training update.  Default: 0.2.
        lam: (float, optional) Hyperparameter for GAE-Lambda calaulation.
            Range: (0, 1).  Default: 0.97
        target_kl: (float, optional) Max KL divergence between new and old
            policies after an update. Used for early stopping. Typically in
            range (0.01, 0.05).  Default: 0.01.
        """
        batch = self.replay.sample()

        for i in range(train_actor_iters):
            self.actor_optimizer.zero_grad()
            loss_pi, approx_kl = actor_loss(
                batch, actor, critic, clip_ratio=clip_ratio, gamma=gamma, lam=lam
            )
            if approx_kl > 1.5 * target_kl:
                break
            loss_pi.backward()
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
        train_actor_iters: int = 80,
        train_critic_iters: int = 80,
        epochs: int = 200,
        steps_per_epoch: int = 4000,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.97,
        target_kl: float = 0.01,
        max_ep_len: int = 1000,
        callbacks: Iterable[Callable] = (),
    ):
        """Proximal Policy Optimization (via objective clipping) with early
        stopping based on approximate KL divergence of the policy network.

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
        train_actor_iters: (int, optional) Max number of actor training steps
            per epoch.  Default: 80.
        train_critic_iters: (int, optional) Max number of critic training steps
            per epoch.  Default: 80.
        clip_ratio: (float, optional) Hyperparameter for clipping in the policy
            objective.  Scales how much the policy is allowed change per
            training update.  Default: 0.2.
        lam: (float, optional) Hyperparameter for GAE-Lambda calaulation.
            Range: (0, 1).  Default: 0.97
        target_kl: (float, optional) Max KL divergence between new and old
            policies after an update. Used for early stopping. Typically in
            range (0.01, 0.05).  Default: 0.01.
        max_ep_len: (int, optional) Maximum length of episode.  Defaults to 1000,
            but *this should be provided for each unique environment!*  This
            has an effect on how end-of-episode rewards are computed.
        callbacks: (Iterable[Callable], optional) callback functions to execute
            at the end of each training epoch.
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
                done = False if ep_length == max_ep_len else done
                self.replay.append([state, action, logprob, reward, done, next_state])
                state = next_state
                ep_reward += reward
                ep_length += 1

                if done or (ep_length == max_ep_len):
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
                train_actor_iters=train_actor_iters,
                train_critic_iters=train_critic_iters,
                clip_ratio=clip_ratio,
                gamma=gamma,
                lam=lam,
                target_kl=target_kl,
            )

            print(f"\r Epoch {epoch} | Avg Reward {self.avg_reward}", end="")

            for callback in callbacks:
                callback(self)
