import gym
import torch

from metis import agents
from metis.trainers import A2C


env = gym.make("CartPole-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env).to(device)
critic = agents.critic(env).to(device)
trainer = A2C(env)

trainer.train(actor, critic)
