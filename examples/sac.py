import gym

from metis import agents
from metis.trainers import SAC


env = gym.make("Pendulum-v0")

actor = agents.actor(env, squashed=True)
critics = [agents.critic(env), agents.critic(env)]
trainer = SAC(env)

trainer.train(actor, critics)
