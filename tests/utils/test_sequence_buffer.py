import numpy as np
from omegaconf import OmegaConf
from dads.utils import EnvStep, EnvSteps, SequenceRB


def _get_conf():
    return OmegaConf.create(
        {
            "capacity": 10000,
            "env": {
                "state_dim": 10,
                "action_dim": 5,
                "skill_dim": 2,
                "skill_continuous": False,
            },
        }
    )


def _get_random_step(env):
    state = np.random.randn(env.state_dim)
    next_state = np.random.randn(env.state_dim)
    action = np.random.randn(env.action_dim)
    reward = 1.0
    done = False
    skill = np.random.randn(env.skill_dim)
    return EnvStep(state, next_state, action, reward, done, skill)


def test_full_sequence_iter():
    sequence_buffer = SequenceRB(_get_conf())

    for _ in range(11):
        sequence_buffer.add_step(_get_random_step(_get_conf().env))
    sequence_buffer.close_sequence()

    for _ in range(15):
        sequence_buffer.add_step(_get_random_step(_get_conf().env))
    sequence_buffer.close_sequence()

    for batch in sequence_buffer.get_full_sequence_iter(10):
        assert len(batch) == 15 or len(batch) == 11


def test_virtual_sequence_iter():
    sequence_buffer = SequenceRB(_get_conf())

    for _ in range(11):
        sequence_buffer.add_step(_get_random_step(_get_conf().env))
    sequence_buffer.close_sequence()

    for _ in range(15):
        sequence_buffer.add_step(_get_random_step(_get_conf().env))
    sequence_buffer.close_sequence()

    for batch in sequence_buffer.get_virtual_sequence_iter(100):
        assert 15 >= len(batch) >= 1
