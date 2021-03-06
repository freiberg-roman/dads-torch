import gym
import numpy as np
from omegaconf import OmegaConf

from dads.env.dads_env import DadsEnvironment
from dads.env.plannable import Plannable


class HalfCheetahEnv(DadsEnvironment, Plannable):
    def __init__(self):
        self.mujoco_env = gym.make("HalfCheetah-v3")
        self._name = "HalfCheetah"

    def get_obs(self, full=True):
        # first entry corresponds to x-position in simulation
        if full:
            return np.concatenate(
                [self.mujoco_env.sim.data.qpos.flat, self.mujoco_env.sim.data.qvel.flat]
            )
        else:
            return np.concatenate(
                [
                    self.mujoco_env.sim.data.qpos.flat[1:],
                    self.mujoco_env.sim.data.qvel.flat,
                ]
            )

    def get_state(self):
        return self.mujoco_env.unwrapped.sim.get_state()

    def set_state(self, state):
        self.mujoco_env.unwrapped.sim.set_state(state)

    def step(self, action):
        _, rew, done, i = self.mujoco_env.step(action)
        return self.get_obs(full=True), rew, done, i

    @property
    def done(self):
        return self.mujoco_env.done

    def reset(self):
        self.mujoco_env.reset()
        return self.get_obs(full=True)

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

    def prep_state(self, state):
        return state[..., 1:]

    def env_reward(self, state, next_state):
        delta = next_state[:, 0] - state[:, 0]
        return delta / self.mujoco_env.dt

    @property
    def name(self):
        return self._name

    @property
    def gym_env(self):
        return self.mujoco_env
