import gym
from torch import nn

from metis import agents
from metis.trainers import TD3


env = gym.make("Pendulum-v0")

actor = agents.actor(env, output_activation=nn.Tanh, deterministic=True)
critics = [agents.critic(env), agents.critic(env)]
trainer = TD3(env)

trainer.train(actor, critics)
