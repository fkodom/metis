
import gym
import torch

from metis import agents, play
from metis.trainers import DQN


env = gym.make("CartPole-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"

dqn = agents.dqn(env).to(device)
trainer = DQN(env)
trainer.train(dqn, epochs=5)

for i in range(100):
    play(env, dqn)
