from abc import ABC, abstractmethod

import numpy as np

from dads.utils.buffer import EnvStep, EnvSteps
from dads.utils.buffer.replay_buffer import ReplayBuffer


class RandomRB(ReplayBuffer):
    def __init__(self, cfg):
        self._cfg = cfg
        self._capacity = 0
        self._ind = 0
        self._s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._next_s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._acts = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._rews = np.empty(cfg.capacity, dtype=np.float32)
        self._dones = np.empty(cfg.capacity, dtype=bool)
        if cfg.env.skill_continuous:
            self._skills = np.empty((cfg.capacity, cfg.env.skill_dim), dtype=np.float32)
        else:
            self._skills = np.empty((cfg.capacity, cfg.env.skill_dim), dtype=np.int32)

    def add(self, state, next_state, action, reward, done, skill):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        self._skills[self._ind, :] = skill

        self._capacity = min(self._capacity + 1, self._cfg.capacity)
        self._ind = (self._ind + 1) % self._cfg.capacity

    def add_batch(self, states, next_states, actions, rewards, dones, skills):
        length_batch = len(states)
        start_ind = self._ind
        end_ind = min(start_ind + length_batch, self._cfg.capacity)
        stored_ind = end_ind - start_ind

        self._s[start_ind:end_ind, :] = states[:stored_ind]
        self._next_s[start_ind:end_ind, :] = next_states[:stored_ind]
        self._acts[start_ind:end_ind, :] = actions[:stored_ind]
        self._rews[start_ind:end_ind] = rewards[:stored_ind]
        self._dones[start_ind:end_ind] = dones[:stored_ind]
        self._skills[start_ind:end_ind, :] = skills[:stored_ind]

        if start_ind + length_batch > self._cfg.capacity:
            self._ind = 0
            self._capacity = self._cfg.capacity
            self.add_batch(
                states[stored_ind:, :],
                next_states[stored_ind:, :],
                actions[stored_ind:, :],
                rewards[stored_ind:],
                dones[stored_ind:],
                skills[stored_ind:, :],
            )
        else:
            self._ind = self._ind + length_batch
            self._capacity = max(self._capacity, self._ind)

    def get_iter(self, it, batch_size):
        return RandomBatchIter(self, it, batch_size)

    def __getitem__(self, item):
        if 0 <= item < len(self):
            return EnvStep(
                self._s[item],
                self._next_s[item],
                self._acts[item],
                self._rews[item],
                self._dones[item],
                self._skills[item],
            )
        else:
            raise ValueError(
                "There are not enough time_steps stored to access this item"
            )

    def __len__(self):
        return self._capacity


class RandomValidationRB(ReplayBuffer):
    def __init__(self, cfg, val_percentage):
        self._train_buffer = RandomRB(cfg)
        self._val_buffer = RandomRB(cfg)
        self._val_percentage = val_percentage

    def __len__(self):
        return len(self._val_buffer) + len(self._train_buffer)

    def get_iter(self, it, batch_size):
        return self._train_buffer.get_iter(it, batch_size), self._val_buffer.get_iter(
            it, batch_size
        )

    def add(self, state, next_state, action, reward, done, skill):
        if (
            len(self._train_buffer) == 0
            or len(self._val_buffer) / len(self) >= self._val_percentage
        ):
            self._train_buffer.add(state, next_state, action, reward, done, skill)
        else:
            self._val_buffer.add(state, next_state, action, reward, done, skill)

    def add_batch(self, states, next_states, actions, rewards, dones, skills):
        if (
            len(self._train_buffer) == 0
            or len(self._val_buffer) / len(self) >= self._val_percentage
        ):
            self._train_buffer.add_batch(
                states, next_states, actions, rewards, dones, skills
            )
        else:
            self._val_buffer.add_batch(
                states, next_states, actions, rewards, dones, skills
            )

    def __getitem__(self, item):
        if item > len(self._train_buffer):
            return self._val_buffer[item - len(self._train_buffer)]
        else:
            return self._train_buffer[item]

    @property
    def train_buffer(self):
        return self._train_buffer

    @property
    def val_buffer(self):
        return self._val_buffer


class RandomBatchIter:
    def __init__(self, buffer: RandomRB, it: int, batch_size: int):
        self._buffer = buffer
        self._it = it
        self._batch_size = batch_size
        self._current_it = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_it < self._it:
            idxs = np.random.randint(0, len(self._buffer), self._batch_size)
            self._current_it += 1
            return EnvSteps(
                self._buffer._s[idxs],
                self._buffer._next_s[idxs],
                self._buffer._acts[idxs],
                self._buffer._rews[idxs],
                self._buffer._dones[idxs],
                self._buffer._skills[idxs],
            )
        else:
            raise StopIteration
