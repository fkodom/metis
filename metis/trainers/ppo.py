from typing import Iterable, Callable

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


def critic_loss(batch, critic: base.Critic, gamma: float = 0.99) -> Tensor:
    states, actions, _, rewards, dones, _ = batch
    returns = torch.zeros_like(rewards)
    returns[:-1] = utils.discount_values(rewards, dones, gamma)[:-1]

    if isinstance(critic, QNetwork):
        values = critic(states)[range(len(actions)), actions.long()]
        return (values - returns.unsqueeze(1)).pow(2).mean()
    else:
        return (critic(states, actions) - returns).pow(2).mean()


class PPO:
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
        actor
        critic
        train_actor_iters: (int) Max number of actor training steps per epoch.
        train_critic_iters: (int) Max number of critic training steps per epoch.
        clip_ratio: (float) Hyperparameter for clipping in the policy objective.
            Scales how much the policy is allowed change per training update.
        gamma: (float) Discount factor. Range: (0, 1)
        lam:: (float) Hyperparameter for GAE-Lambda calaulation. Range: (0, 1)
        target_kl: (float) Max KL divergence between new and old policies after
            an update. Used for early stopping. Typically in range (0.01, 0.05)
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
        max_ep_len: int = 1000,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.97,
        target_kl: float = 0.01,
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
        train_actor_iters: (int) Max number of actor training steps per epoch.
        train_critic_iters: (int) Max number of critic training steps per epoch.
        epochs: (int) Number of training epochs (number of policy updates)
        steps_per_epoch: (int) Number of environment steps (or turns) per epoch
        max_ep_len: (int) Max length of an environment episode (or game)
        clip_ratio: (float) Hyperparameter for clipping in the policy objective.
            Scales how much the policy is allowed change per training update.
        gamma: (float) Discount factor. Range: (0, 1)
        lam: (float) Hyperparameter for GAE-Lambda calaulation. Range: (0, 1)
        target_kl: (float) Max KL divergence between new and old policies after
            an update. Used for early stopping. Typically in range (0.01, 0.05)
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
