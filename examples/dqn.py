import argparse

import gym
import torch

from metis import agents, play
from metis import trainers


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v1")
parser.add_argument("--trainer", default="DQN")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

env = gym.make(args.env)
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

trainer = getattr(trainers, args.trainer)(env)
actor = agents.default_dqn(env, trainer)
trainer.train(actor)

for i in range(100):
    play(env, actor)
