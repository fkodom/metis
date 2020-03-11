<div align="center">
    <img width="525" height="425" src="https://upload.wikimedia.org/wikipedia/commons/d/dc/9_Metis_symbol.svg">
</div>

# Metis

Metis is a minimalist library for training RL agents in PyTorch.  It implements 
many common training algorithms, with a focus on actor-critic methods. 
Includes SAC, TD3, PPO, A2C, VPG.

### Why the name 'metis'?
The meaning is three-fold: 
1. The Greek word *metis* meant a quality that combined wisdom and cunning.  Metis feels like an apt description for the goal of RL -- find a cunning way to gain wisdom about a particular environment.
2. *Metis* was also a Titaness of Greek mythology, known as the embodiment of "prudence", "wisdom", or "wise counsel".  Again, sounds like a good description for what RL aspires to be.
3. "Metis" sounds vaguely similar to "meta", as in meta-learning.  For those out there (which definitely includes myself) who need something simpler to remember.

**NOTE:** I've been told that "metis" is actually pronounced (*mee-tis*), which
kind of squashes the third interpretation.  But I imagine that some folks will 
still pronounce it (*meh-tis*) like I do.

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

### Organization
Motivated by goals (1) and (2) above, each training algorithm is completely 
independent of the others.  They do not inherit methods from any parent class, 
and each independently defines its own update and loss functions.  At times, 
this might seem wasteful, because significant amounts of code are repeated.
We certainly could define (semi-)generic parent classes for on-policy and 
off-policy trainers (or for all generic trainers), which might make the code 
less redundant.  In practice, however, RL algorithms are difficult to write in 
a completely agnostic way.  We would need to create additional class methods to 
handle the differences between algorithms (e.g. number of critic networks, 
number of target networks, rules for updates, etc.), which would reduce the 
readability and hackability of the code.

For the above reasons, relatively few abstractions are used (e.g. parent classes,
abstract methods), which makes the code as explicit as possible.  I believe this 
makes it more usable for real-world applications.  In reality, I expect users to
extract the bits and pieces they need, and adapt them to new use cases.  I'm not 
sure it would be possible to write an RL library generic enough for every use
case -- or at the very least, I'm not clever enough to do it.  As I tell myself 
many days, "Keep it simple, stupid."


## Getting Started
Metis tries to be as user-friendly as possible, without reducing hackability of
the overall project.  Training your first RL agent can be done in just a few lines
of code:

```python
import gym
from metis import agents
from metis.trainers import PPO

env = gym.make("Pendulum-v0")
# Create generic actor/critic modules for the given environment.
actor = agents.actor(env)
critic = agents.critic(env)
trainer = PPO(env)
trainer.train(actor, critic)
```

GPU execution is also supported out-of-the-box.  Simply push your RL agents to the
desired device, and the trainer will handle the rest.

```python
actor.cuda()
critic.cuda()
trainer.train(actor, critic)
```

Training on multiple GPUs is only slightly more work.  We use `DataParallel`
from the PyTorch API to specify which devices to run on.  Again, there are no 
changes needed for the trainer object.

```python
from torch.nn import DataParallel

# Assumes that two GPUs are available with device IDs: 0, 1
dp_actor = DataParallel(actor, device_ids=[0, 1])
dp_critic = DataParallel(critic, device_ids=[0, 1])
trainer.train(dp_actor, dp_critic)
```

Finally, all policies that derive from `metis.agents.Actor` can be visualized
using the `metis.play` method.  A game window will be constructed, and the agent
interacts with the environment until a `done` flag is encountered.

```python
from metis import play

play(env, actor)
```

### Examples
In addition to the code snippets above, several example training scripts are 
included in the `examples` folder.  They are very minimal and don't involve any 
callbacks.  


## Algorithms
| Name                                                   | Discrete | Continuous | Experience Replay | 
|--------------------------------------------------------|----------|------------|-------------------|
| SAC:  Soft Actor-Critic                                | &#9745;  | &#9745;    | &#9745;           |
| TD3:  Twin-Delayed Deep Deterministic Policy Gradients |          | &#9745;    | &#9745;           |
| DDPG:  Deep Deterministic Policy Gradients             |          | &#9745;    | &#9745;           |
| PPO:  Proximal Policy Optimization                     | &#9745;  | &#9745;    |                   |
| A2C:  Advantage Actor-Critic                           | &#9745;  | &#9745;    |                   |
| VPG:  Vanilla Policy Gradients                         | &#9745;  | &#9745;    |                   |

**NOTE:**  The definition of VPG here differs from some other APIs.  Others 
(e.g. `spinup`) define VPG using both an actor and critic network, which is 
equivalent to A2C in this repository.  Our VPG does *not* contain a critic 
network, since this aligns more closely with the original 
[REINFORCE paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf').


## Future Work
* Add documentation on how to modify existing trainers or build your own
* Add more callback functions for logging training info, early stopping, etc.
* Add documentation to README on defining custom actors/critics
