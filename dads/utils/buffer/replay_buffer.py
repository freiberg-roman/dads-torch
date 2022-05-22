from abc import ABC, abstractmethod

from dads.utils import EnvStep, EnvSteps


class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, state, next_state, action, reward, done, skill):
        pass

    def add_step(self, time_step: EnvStep):
        self.add(
            time_step.state,
            time_step.next_state,
            time_step.action,
            time_step.reward,
            time_step.done,
            time_step.skill,
        )

    @abstractmethod
    def add_batch(self, states, next_states, actions, rewards, dones, skills):
        pass

    def add_batch_steps(self, time_steps: EnvSteps):
        self.add_batch(
            time_steps.states,
            time_steps.next_states,
            time_steps.actions,
            time_steps.rewards,
            time_steps.dones,
            time_steps.skills,
        )

    @abstractmethod
    def get_iter(self, it, batch_size):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        return 0
