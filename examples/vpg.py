import gym
import torch

from metis import agents
from metis.trainers import VPG


env = gym.make("CartPole-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env).to(device)
trainer = VPG(env)

trainer.train(actor)
