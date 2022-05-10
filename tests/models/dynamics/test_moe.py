import torch
from dads.models import MixtureOfExperts
from torch.optim import Adam


def test_simple_forward():
    skill_model = MixtureOfExperts(state_dim=5, skill_dim=4)

    # create data with batch size 100
    states = torch.randn(100, 5)
    skills = torch.randn(100, 4)

    pred_dist = skill_model.forward(states, skills)
    sample = pred_dist.sample((10,))
    assert sample.shape == torch.Size([10, 100, 5])


def test_compute_log_prob():
    skill_model = MixtureOfExperts(state_dim=10, skill_dim=3)

    # create data with batch size 100
    states = torch.randn(100, 10)
    next_states = torch.randn(100, 10)
    skills = torch.randn(100, 3)

    log_prob = skill_model.log_prob(states, skills, next_states)
    assert log_prob.shape == torch.Size([100])


def test_simple_backprop():
    skill_model = MixtureOfExperts(state_dim=3, skill_dim=18)
    optimizer = Adam(skill_model.parameters(), lr=0.0003)

    # create data with batch size 100
    states = torch.randn(100, 3)
    next_states = torch.randn(100, 3)
    skills = torch.randn(100, 18)

    # perform one optimization step
    optimizer.zero_grad()
    skill_model.loss(states, skills, next_states).backward()
    optimizer.step()


def test_pred_next_state_deterministic():
    skill_model = MixtureOfExperts(state_dim=3, skill_dim=18)

    # create data with batch size 100
    states = torch.randn(100, 3)
    skills = torch.randn(100, 18)

    next_states = skill_model.pred_next(states, skills, deterministic=True)
    assert next_states.shape == torch.Size([100, 3])


def test_pred_next_state_random():
    skill_model = MixtureOfExperts(state_dim=3, skill_dim=18)

    # create data with batch size 100
    states = torch.randn(100, 3)
    skills = torch.randn(100, 18)

    next_states = skill_model.pred_next(states, skills)
    assert next_states.shape == torch.Size([1, 100, 3])
