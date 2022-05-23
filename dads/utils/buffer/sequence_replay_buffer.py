import random
import warnings

import numpy as np

from dads.utils.buffer.replay_buffer import EnvSteps, ReplayBuffer


class SequenceRB(ReplayBuffer):
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

        self._seq = []
        self._valid_seq = []

    def add(self, state, next_state, action, reward, done, skill):
        self._seq.append((state, next_state, action, reward, done, skill))

    def add_batch(self, states, next_states, actions, rewards, dones, skills):
        for state, next_state, action, reward, done, skill in zip(
            states, next_states, actions, rewards, dones, skills
        ):
            self.add(state, next_state, action, reward, done, skill)

    def close_sequence(self):
        len_seq = len(self._seq)
        if len_seq > self._cfg.capacity:
            warnings.warn("Sequence will be discarded since it exceeds buffer capacity")
            self._seq = []
            return

        if len_seq > self._cfg.capacity - self._ind + 1:
            self._ind = 0
        valid_start = self._ind
        valid_end = valid_start + len_seq

        self._remove_overlapping_seqs((valid_start, valid_end))
        self._valid_seq.append((valid_start, valid_end))

        for state, next_state, action, reward, done, skill in self._seq:
            self._s[self._ind, :] = state
            self._next_s[self._ind, :] = next_state
            self._acts[self._ind, :] = action
            self._rews[self._ind] = reward
            self._dones[self._ind] = done
            self._skills[self._ind, :] = skill
            self._ind += 1

        assert self._ind == valid_end
        self._capacity += len_seq
        self._seq = []

    def get_iter(self, it, batch_size):
        return self.get_full_sequence_iter(it)

    def get_full_sequence_iter(self, it):
        """
        Returns sequences according to 'close_sequence' calls.
        All sequences are equally likely to be chosen.
        """
        return FullSequenceIter(self, it)

    def get_virtual_sequence_iter(self, it):
        """
        Returns sequences that can start anywhere, but are still ending
        according to 'close_sequence' calls.
        """
        return VirtualSequenceIter(self, it)

    def get_true_k_sequence_iter(self, it, k):
        """
        Returns k-step sequences which are samples from steps from one sequence
        """
        return TrueKSequenceIter(self, it, k)

    def _remove_overlapping_seqs(self, seq_boundaries):
        start, end = seq_boundaries
        self._valid_seq.sort(key=lambda x: x[0])  # sort after valid starts
        while True:
            ise = next(
                (
                    (i, se[0], se[1])
                    for i, se in enumerate(self.valid_seq)
                    if se[1] > start
                ),
                None,
            )
            if ise is None or ise[1] >= end:
                return
            else:
                self._valid_seq.remove((ise[1], ise[2]))
                self._capacity -= ise[2] - ise[1]

    def __getitem__(self, item):
        s, e = self._valid_seq[item]
        return EnvSteps(
            self._s[s:e],
            self._next_s[s:e],
            self._acts[s:e],
            self._rews[s:e],
            self._dones[s:e],
            self._skills[s:e],
        )

    def __len__(self):
        return self._capacity

    @property
    def valid_seq(self):
        return self._valid_seq

    @property
    def states(self):
        return self._s

    @property
    def next_states(self):
        return self._next_s

    @property
    def actions(self):
        return self._acts

    @property
    def rewards(self):
        return self._rews

    @property
    def dones(self):
        return self._dones

    @property
    def skills(self):
        return self._skills

    @property
    def capacity(self):
        return self._capacity


class FullSequenceIter:
    def __init__(self, buffer: SequenceRB, it: int):
        self._buffer = buffer
        self._it = it
        self._current_it = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_it < self._it:
            start, end = random.choice(self._buffer.valid_seq)
            self._current_it += 1
            return EnvSteps(
                self._buffer.states[start:end],
                self._buffer.next_states[start:end],
                self._buffer.actions[start:end],
                self._buffer.rewards[start:end],
                self._buffer.dones[start:end],
                self._buffer.skills[start:end],
            )
        else:
            raise StopIteration


class VirtualSequenceIter:
    def __init__(self, buffer: SequenceRB, it: int):
        self._buffer = buffer
        self._it = it
        self._current_it = 0
        self.weights = np.array(
            [(e - s) for s, e in buffer.valid_seq], dtype=np.float64
        )
        self.weights /= np.sum(self.weights)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_it < self._it:
            ind = np.random.choice(
                np.arange(len(self._buffer.valid_seq)), p=self.weights
            )
            s, e = self._buffer.valid_seq[ind]
            s_from = np.random.randint(s, e)
            self._current_it += 1
            return EnvSteps(
                self._buffer.states[s_from:e],
                self._buffer.next_states[s_from:e],
                self._buffer.actions[s_from:e],
                self._buffer.rewards[s_from:e],
                self._buffer.dones[s_from:e],
                self._buffer.skills[s_from:e],
            )
        else:
            raise StopIteration


class TrueKSequenceIter:
    def __init__(self, buffer: SequenceRB, it: int):
        self._buffer = buffer
        self._it = it
        self._current_it = 0

    def __iter__(self):
        pass

    def __next__(self):
        pass
