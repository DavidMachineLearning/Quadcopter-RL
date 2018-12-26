# Quadcopter-RL

This project is design to teach a quadcopter how to fly using Deep Reinforcement Learning. I used an environment provided by www.udacity.com, included in "physics_sim.py", it doesn't provide a Virtual 3D environment but it also doesn't need any configuration. In case you want to see the drone on a 3D environment you can follow up the instructions at https://github.com/udacity/RL-Quadcopter. For this project I'll use a reinforcement learning algorithm called Deep Deterministic Policy Gradients (DDPG) using Tensorflow. I also used Ornstein-Uhlenbeck process to implement some "noise" to the output of the model, using it to help the agent exploring the environment. You can read more about it at https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process .

### Files

- Tasks.py: File where all tasks are defined.
- ActorCritic.py: File containing the structure for actor's model and critic's model.
- Helpers.py: File containing helper functions like "Replay" and "Ornstein-Uhlenbeck process".
- agent.py: File where is defined the agent.
- physics_sim.py: This file contains the simulator for the quadcopter. DO NOT MODIFY THIS FILE.
