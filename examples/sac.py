import gym
import torch

from metis import agents, play
from metis.trainers import SAC


env = gym.make("Pendulum-v0")
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = agents.actor(env, hidden_sizes=(256, 256), squashed=True).to(device)
critics = [
    agents.critic(env, hidden_sizes=(256, 256)).to(device),
    agents.critic(env, hidden_sizes=(256, 256)).to(device),
]
trainer = SAC(env)
trainer.train(actor, critics)

for i in range(100):
    play(env, actor)
