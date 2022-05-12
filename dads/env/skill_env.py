import numpy as np
from omegaconf.omegaconf import OmegaConf

from .dads_env import DadsEnvironment


class SkillEnvironment:
    def __init__(
        self,
        env: DadsEnvironment,
        env_conf,
        skill_dim=2,
        skill_continuous=True,
        skill_sampling="uniform",
    ):
        self._env = env
        self._env_conf: OmegaConf = env_conf
        self._skill_dim = skill_dim
        self._skill_type = skill_continuous
        self._skill_sampling = skill_sampling
        self._skill = self.gen_skill()

        self._env_conf.skill_continuous = skill_continuous
        self._env_conf.skill_dim = skill_dim
        self._total_steps = 0

    def gen_skill(self):
        if self._skill_type == "continues":
            return np.random.uniform(-1, 1, (self._skill_dim,))
        else:
            return np.random.uniform(-1, 1, (self._skill_dim,))

    def get_state(self):
        return self._env.get_state()

    def set_state(self, state):
        self._env.set_state(state)

    def step(self, action):
        self._total_steps += 1
        return *self._env.step(action), self._skill

    def sample_skills(self, n):
        if self._skill_type == "continues":
            return np.random.uniform(-1, 1, (n, self._skill_dim))
        else:
            return np.random.uniform(-1, 1, (n, self._skill_dim))

    @property
    def done(self):
        return self._env.done

    @property
    def total_steps(self):
        return self._total_steps

    def prep_state(self):
        return self._env.prep_state

    def reset(self):
        self._skill = self.gen_skill()
        return self._env.reset(), self._skill

    def get_env_cfg(self):
        return self._env_conf
