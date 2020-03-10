import gym
import torch
from torch import nn

from metis import agents, play
from metis.trainers import TD3


env = gym.make("Pendulum-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(
    env,
    hidden_sizes=(256, 256),
    output_activation=nn.Tanh(),
    deterministic=True,
).to(device)
critics = [
    agents.critic(env, hidden_sizes=(256, 256)).to(device),
    agents.critic(env, hidden_sizes=(256, 256)).to(device),
]
trainer = TD3(env)
trainer.train(actor, critics)

for i in range(100):
    play(env, actor)
