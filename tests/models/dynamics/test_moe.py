import torch
from dads.models import MixtureOfExperts
from torch.optim import Adam
from omegaconf.omegaconf import OmegaConf

cfg_model = OmegaConf.create(
    {
        "name": "MixtureOfExperts",
        "device": "cpu",
        "hidden_dim": 512,
        "num_experts": 4,
        "lr": 0.0003,
        "env": {"state_dim": 5, "skill_dim": 4, "num_coordinates": 1},
    }
)


def test_simple_forward():
    skill_model = MixtureOfExperts(cfg_model, prep_input_fn=lambda x: x[..., 1:])

    # create data with batch size 100
    states = torch.randn(100, 5)
    skills = torch.randn(100, 4)

    pred_dist = skill_model.forward(states, skills)
    sample = pred_dist.sample((10,))
    assert sample.shape == torch.Size([10, 100, 5])


def test_compute_log_prob():
    cfg_model.env.state_dim = 10
    cfg_model.env.skill_dim = 3
    skill_model = MixtureOfExperts(cfg_model, prep_input_fn=lambda x: x[..., 1:])

    # create data with batch size 100
    states = torch.randn(100, 10)
    next_states = torch.randn(100, 10)
    skills = torch.randn(100, 3)

    log_prob = skill_model.log_prob(states, skills, next_states)
    assert log_prob.shape == torch.Size([100])


def test_simple_backprop():
    cfg_model.env.state_dim = 3
    cfg_model.env.skill_dim = 18
    cfg_model.env.num_coordinates = 0
    skill_model = MixtureOfExperts(cfg_model)
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
    cfg_model.env.state_dim = 3
    cfg_model.env.skill_dim = 18
    cfg_model.env.num_coordinates = 0
    skill_model = MixtureOfExperts(cfg_model)

    # create data with batch size 100
    states = torch.randn(100, 3)
    skills = torch.randn(100, 18)

    next_states = skill_model.pred_next(states, skills, deterministic=True)
    assert next_states.shape == torch.Size([100, 3])


def test_pred_next_state_random():
    cfg_model.env.state_dim = 3
    cfg_model.env.skill_dim = 18
    cfg_model.env.num_coordinates = 0
    skill_model = MixtureOfExperts(cfg_model)

    # create data with batch size 100
    states = torch.randn(100, 3)
    skills = torch.randn(100, 18)

    next_states = skill_model.pred_next(states, skills)
    assert next_states.shape == torch.Size([1, 100, 3])
