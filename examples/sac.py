import gym
import torch

from metis import agents
from metis.trainers import SAC


env = gym.make("Pendulum-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env, squashed=True).to(device)
critics = [agents.critic(env).to(device), agents.critic(env).to(device)]
trainer = SAC(env)

trainer.train(actor, critics)
