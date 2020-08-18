import argparse

import gym
import torch

from metis import agents, play
from metis import trainers
from metis.trainers import SAC


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v1")
parser.add_argument("--trainer", default="PPO")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

env = gym.make(args.env)
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

trainer = getattr(trainers, args.trainer)(env)
actor = agents.default_actor(env, trainer).to(device)
critics = [
    agents.default_critic(env, trainer).to(device), 
    agents.default_critic(env, trainer).to(device),
]
trainer.train(actor, critics)

for i in range(100):
    play(env, actor)
