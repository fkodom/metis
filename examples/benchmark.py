import gym
import torch
import matplotlib.pyplot as plt

import metis
from metis.agents import actor, critic
from metis.trainers import SAC
from metis.utils import smooth_values


def train(seed: int):
    metis.seed(seed)
    env = gym.make("Pendulum-v0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SAC(env)
    trainer.train(
        actor=actor(env, hidden_sizes=(256, 256), squashed=True).to(device),
        critics=[
            critic(env, hidden_sizes=(256, 256)).to(device),
            critic(env, hidden_sizes=(256, 256)).to(device),
        ],
        epochs=5,
    )

    return trainer.ep_rewards


def benchmark():
    seeds = list(range(5))
    rewards = [train(seed) for seed in seeds]
    rewards = torch.as_tensor(rewards, dtype=torch.float)

    rewards = smooth_values(rewards, window=5)
    rewards_mean = rewards.mean(dim=0)
    rewards_std = rewards.std(dim=0)

    return rewards_mean, rewards_std


# Carefully check these final steps!  If there are any bugs, then you'll have to
# repeat your benchmark programs.
#
# Compute mean and standard deviation (already smoothed over epochs above)
mean, std = benchmark()
mean, std = mean.numpy(), std.numpy()
steps = range(mean.size)

# Plot the results.  In general, you should also save these to file, but we'll
# avoid creating unnecessary files in this demo.
plt.plot(steps, mean.tolist(), color="blue")
plt.fill_between(steps, mean - std, mean + std, color="blue", alpha=0.2)
plt.show()
