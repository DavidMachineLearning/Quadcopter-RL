import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Main class that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose, init_velocities, init_angle_velocities, runtime, target_pos):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        # Environment
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Target position
        self.target_pos = target_pos

    def _step(self, rotor_speeds, reward_func):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += reward_func()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
                 

class Takeoff(Task):
    """Class specific for the takeoff task."""
    def __init__(self, init_pose=np.array([0., 0., 0., 0., 0., 0.]),
                 init_velocities=np.array([0., 0., 0.]),
                 init_angle_velocities=np.array([0., 0., 0.]) ,
                 runtime=5., target_pos=np.array([0., 0., 10.])):
                 super().__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)

    def reward_func(self):
        """Uses current pose of sim to return reward."""
        reward = np.tanh(1.-.3*(abs(self.sim.pose[2] - self.target_pos[2])).sum())
        return reward

    def step(self, rotor_speeds):
        return self._step(rotor_speeds, self.reward_func)


class Hover(Task):
    """Class specific for the hovering task."""
    def __init__(self, init_pose=np.array([0., 0., 10., 0., 0., 0.]),
                 init_velocities=np.array([0., 0., 0.]),
                 init_angle_velocities=np.array([0., 0., 0.]) ,
                 runtime=5., target_pos=np.array([0., 0., 10.])):
                 super().__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)

    def reward_func(self):
        """Uses current pose of sim to return reward."""
        reward = np.tanh(1.-.003*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        return reward

    def step(self, rotor_speeds):
        return self._step(rotor_speeds, self.reward_func)


class Land(Task):
    """Class specific for the landing task."""
    def __init__(self, init_pose=np.array([0., 0., 10., 0., 0., 0.]),
                 init_velocities=np.array([0., 0., 0.]),
                 init_angle_velocities=np.array([0., 0., 0.]) ,
                 runtime=5., target_pos=np.array([0., 0., 0.])):
                 super().__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)

    def reward_func(self):
        """Uses current pose of sim to return reward."""
        z_err = .003*(abs(self.sim.pose[2] - self.target_pos[2])).sum()
        angle_err = .0005*(abs(self.sim.pose[3:] - np.array([0., 0., 0.]))).sum()
        reward = np.tanh(1. - z_err - angle_err)
        return reward

    def step(self, rotor_speeds):
        return self._step(rotor_speeds, self.reward_func)


class Flyto(Task):
    """Class specific for defining the task of fly to a specific point."""
    def __init__(self, init_pose=np.array([0., 0., 10., 0., 0., 0.]),
                 init_velocities=np.array([0., 0., 0.]),
                 init_angle_velocities=np.array([0., 0., 0.]) ,
                 runtime=5., target_pos=np.array([10., 10., 15.])):
                 super().__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)

    def reward_func(self):
        """Uses current pose of sim to return reward."""
        reward = np.tanh(1.-.003*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        return reward

    def step(self, rotor_speeds):
        return self._step(rotor_speeds, self.reward_func)
