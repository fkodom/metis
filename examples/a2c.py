import gym

from metis import agents
from metis.trainers import A2C


env = gym.make("CartPole-v1")

actor = agents.actor(env)
critic = agents.critic(env)
trainer = A2C(env)

trainer.train(actor, critic)
