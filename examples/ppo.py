import gym

from metis import agents
from metis.trainers import PPO

env = gym.make("Pendulum-v0")

actor = agents.actor(env)
critic = agents.critic(env)
trainer = PPO(env)

trainer.fit(actor, critic)
