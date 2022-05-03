from abc import ABC, abstractmethod


class Plannable(ABC):
    @abstractmethod
    def sim_steps(self, init_state, actions):
        pass
