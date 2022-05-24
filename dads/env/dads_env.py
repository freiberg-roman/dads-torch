from abc import ABC, abstractmethod


class DadsEnvironment(ABC):
    def __init__(self):
        self._state = None

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self):
        pass

    @abstractmethod
    def get_obs(self, full=True):
        pass

    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def done(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def gym_env(self):
        pass

    @abstractmethod
    def env_reward(self, state, next_state):
        pass

    def __enter__(self):
        self._state = self.get_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.set_state(self._state)
