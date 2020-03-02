import gym
import torch
from torch import nn

from metis import agents
from metis.trainers import TD3


env = gym.make("Pendulum-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env, output_activation=nn.Tanh, deterministic=True).to(device)
critics = [agents.critic(env).to(device), agents.critic(env).to(device)]
trainer = TD3(env)

trainer.train(actor, critics)
