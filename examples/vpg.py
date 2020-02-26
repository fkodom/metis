import gym

from metis import agents
from metis.trainers import VPG


env = gym.make("CartPole-v1")

actor = agents.actor(env)
trainer = VPG(env)

trainer.fit(actor)
