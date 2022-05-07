import numpy as np
from dads.utils import RandomRB, EnvStep, EnvSteps
from omegaconf import OmegaConf


def create_simple_buffer():
    cfg = {
        "capacity": 10000,
        "env": {
            "state_dim": 4,
            "action_dim": 1,
            "skill_dimensions": 6,
            "skill_continuous": True,
        },
    }
    return RandomRB(OmegaConf.create(cfg))


def test_random_rb_simple_index():
    cfg = {
        "capacity": 10000,
        "env": {
            "state_dim": 3,
            "action_dim": 2,
            "skill_dimensions": 5,
            "skill_continuous": True,
        },
    }
    state = np.random.randn(3)
    next_state = np.random.randn(3)
    action = np.random.randn(2)
    reward = 1.0
    done = False
    skill = np.random.randn(5)

    buffer = RandomRB(OmegaConf.create(cfg))
    buffer.add(state, next_state, action, reward, done, skill)
    buffer.add_step(EnvStep(state, next_state, action, reward, done, skill))

    assert len(buffer) == 2
    assert np.array_equal(buffer[0].state, state.astype(np.float32))


def test_random_rb_batch_add():
    buffer = create_simple_buffer()

    states = np.random.randn(10, 4)
    next_states = np.random.randn(10, 4)
    actions = np.random.randn(10, 1)
    rewards = np.random.randn(10)
    dones = np.array([False] * 10)
    skills = np.random.randn(10, 6)

    buffer.add_batch(states, next_states, actions, rewards, dones, skills)
    buffer.add_batch_steps(
        EnvSteps(states, next_states, actions, rewards, dones, skills)
    )

    assert len(buffer) == 20
    assert np.array_equal(buffer[9].state, states.astype(np.float32)[9])


def test_random_rb_overflow():
    cfg = {
        "capacity": 9,
        "env": {
            "state_dim": 4,
            "action_dim": 1,
            "skill_dimensions": 6,
            "skill_continuous": True,
        },
    }
    rbuffer = RandomRB(OmegaConf.create(cfg))

    states = np.random.randn(10, 4)
    next_states = np.random.randn(10, 4)
    actions = np.random.randn(10, 1)
    rewards = np.array([1.0] * 10)
    dones = np.array([False] * 10)
    skills = np.random.randn(10, 6)

    rbuffer.add_batch(states, next_states, actions, rewards, dones, skills)
    assert len(rbuffer) == 9
    assert np.array_equal(rbuffer[8].state, states.astype(np.float32)[8])
    assert np.array_equal(rbuffer[0].state, states.astype(np.float32)[9])
    assert np.array_equal(rbuffer[1].state, states.astype(np.float32)[1])


def test_random_rb_iter():
    buffer = create_simple_buffer()

    states = np.random.randn(10, 4)
    next_states = np.random.randn(10, 4)
    actions = np.random.randn(10, 1)
    rewards = np.array([1.0] * 10)
    dones = np.array([False] * 10)
    skills = np.random.randn(10, 6)

    buffer.add_batch(states, next_states, actions, rewards, dones, skills)

    for it, batch in enumerate(buffer.get_iter(it=5, batch_size=3)):
        iter = it + 1
        assert len(batch) == 3

    assert iter == 5
