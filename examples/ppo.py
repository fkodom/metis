import gym
import torch

from metis import agents
from metis.trainers import PPO

env = gym.make("Pendulum-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env).to(device)
critic = agents.critic(env).to(device)
trainer = PPO(env)

trainer.train(actor, critic)
