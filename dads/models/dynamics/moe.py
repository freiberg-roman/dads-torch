import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal

from dads.utils import EnvStep


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        cfg,
        prep_input_fn=lambda x: x,
    ):
        super(MixtureOfExperts, self).__init__()

        self.bn_in = nn.BatchNorm1d(cfg.env.state_dim - cfg.env.num_coordinates)
        self.bb_out = nn.BatchNorm1d(cfg.env.state_dim)
        self._prep_input_fn = prep_input_fn

        self.linear1 = nn.Linear(
            cfg.env.state_dim + cfg.env.skill_dim - cfg.env.num_coordinates,
            cfg.hidden_dim,
        )
        self.linear2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.expert_log_prob_contribution = nn.Linear(cfg.hidden_dim, cfg.num_experts)
        self.expert_heads = []

        for _ in range(cfg.num_experts):
            self.expert_heads.append(nn.Linear(cfg.hidden_dim, cfg.env.state_dim))

        self._skill_dim = cfg.env.skill_dim

    def forward(self, state, skill):
        n_state = self.bn_in(self._prep_input_fn(state))
        net_in = torch.cat([n_state, skill], 1)

        x1 = F.silu(self.linear1(net_in))
        x2 = F.silu(self.linear2(x1))

        expert_log_prob = self.expert_log_prob_contribution(x2)
        categorical_experts = Categorical(logits=expert_log_prob)

        means = []
        for head in self.expert_heads:
            means.append(head(x2))
        means = torch.stack(means, 1)

        ind_expert_dist = Independent(
            MultivariateNormal(means, torch.eye(means.size(dim=-1))), 0
        )
        return MixtureSameFamily(categorical_experts, ind_expert_dist)

    def log_prob(self, state, skill, next_state):
        n_next_state_delta = self.bb_out(next_state - state)
        prob = self.forward(state, skill)
        return prob.log_prob(n_next_state_delta)

    def loss(self, state, skill, next_state):
        log_prob = self.log_prob(state, skill, next_state)
        return -log_prob.mean()  # NLL

    @torch.no_grad()
    def pred_next(self, state, skill, sample_n=1, deterministic=False):
        if not deterministic:
            pred_delta = self.forward(state, skill).sample((sample_n,))
        else:
            pred_delta = self.forward(state, skill).mean

        # denorm pred_delta
        pred_delta = (
            pred_delta * torch.sqrt(self.bb_out.running_var + self.bb_out.eps)
            + self.bb_out.running_mean
        )
        return state + pred_delta

    @torch.no_grad()
    def intrinsic_rew(self, step: EnvStep, skill_samples):

        state, next_state, _, _, _, skill = step.to_torch_batch()
        skill_samples = torch.from_numpy(skill_samples).to(torch.float32)

        self.eval()
        with torch.no_grad():
            model_log_prob = self.log_prob(state, skill, next_state)

        L = len(skill_samples)
        state = state.repeat(L, 1)
        next_state = next_state.repeat(L, 1)
        with torch.no_grad():
            model_prob_samples = (
                self.log_prob(state, skill_samples, next_state).exp().sum(axis=0)
            )
        self.train()
        return (
            model_log_prob - torch.log(model_prob_samples) + torch.log(torch.tensor(L))
        ).item()

    def update_parameters(self, batch, optimizer: torch.optim.Optimizer):
        states, next_states, _, _, _, skills = batch.to_torch_batch()
        loss = self.loss(states, skills, next_states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def to(self):
        pass  # TODO
