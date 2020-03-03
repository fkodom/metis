import gym
import torch
from torch import nn

from metis import agents
from metis.trainers import DDPG


env = gym.make("Pendulum-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env, output_activation=nn.Tanh(), deterministic=True).to(device)
critic = agents.critic(env).to(device)
trainer = DDPG(env)

trainer.train(actor, critic)
