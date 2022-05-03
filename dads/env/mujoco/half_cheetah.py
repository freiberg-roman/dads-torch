import gym
import numpy as np

from dads.env.dads_env import DadsEnvironment
from dads.env.plannable import Plannable


class HalfCheetahEnv(DadsEnvironment, Plannable):
    def __init__(self):
        self.mujoco_env = gym.make("HalfCheetah-v2")

    def get_obs(self):
        # first entry corresponds to x-position in simulation
        return np.concatenate(
            [self.mujoco_env.sim.data.qpos.flat[1:], self.mujoco_env.sim.data.qvel.flat]
        )

    def get_state(self):
        return self.mujoco_env.unwrapped.sim.get_state()

    def set_state(self, state):
        self.mujoco_env.unwrapped.sim.set_state(state)

    def step(self, action):
        return self.mujoco_env.step(action)

    @property
    def done(self):
        return self.mujoco_env.done

    def reset(self):
        self.mujoco_env.reset()

    def sim_steps(self, init_state, actions):
        rew_sum = 0

        with self as env:
            env.set_state(init_state)

            for act in actions:
                state, reward, done, _ = self.step(act)

                if done:
                    break
                rew_sum += reward
        return state, rew_sum
