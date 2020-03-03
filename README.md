# Metis

Metis is a minimalist library for training RL agents in PyTorch.  It implements 
many common training algorithms, with a focus on actor-critic methods. 
Includes SAC, TD3, PPO, A2C, VPG.

![metis-symbol](https://raw.githubusercontent.com/fkodom/metis/master/docs/media/metis.jpg)


### Why the name 'metis'?

The meaning is three-fold: 
1. The Greek word *metis* meant a quality that combined wisdom and cunning.  Metis feels like an apt description for the goal of RL -- find a cunning way to gain wisdom about a particular environment.
2. *Metis* was also a Titaness of Greek mythology, known as the embodiment of "prudence", "wisdom", or "wise counsel".  Again, sounds like a good description for what RL aspires to be.
3. "Metis" sounds vaguely similar to "meta", as in meta-learning.  For those out there (which definitely includes myself) who need something simpler to remember.


### Philosophy

There are lots of RL libraries out there.  In my experience, many of them are 
unnecessarily complicated, which makes them a nightmare to use.  Others are much
nicer (e.g. OpenAI's [spinningup](https://github.com/openai/spinningup)), but they
are not designed for general engineering applications -- they are not so easily 
"hackable".  Metis was started as a personal project, with the goal of creating 
a general-purpose RL library that is easy to use and understand (as much as is
possible for RL algorithms).

Guiding development goals, in order of importance:
1.  Usability
2.  Hackability
3.  Simplicity
4.  Efficiency
