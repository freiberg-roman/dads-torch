import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal


class MixtureOfExperts(nn.Module):
    def __init__(self, state_dim, skill_dim, hidden_dim=1024, num_experts=4):
        super(MixtureOfExperts, self).__init__()

        self.bn_in = nn.BatchNorm1d(state_dim)
        self.bb_out = nn.BatchNorm1d(state_dim)

        self.linear1 = nn.Linear(state_dim + skill_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.expert_log_prob_contribution = nn.Linear(hidden_dim, num_experts)
        self.expert_heads = []

        for _ in range(num_experts):
            self.expert_heads.append(nn.Linear(hidden_dim, state_dim))

        self._skill_dim = skill_dim

    def forward(self, state, skill):
        n_state = self.bn_in(state)
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
            MultivariateNormal(means, torch.eye(means.size())), 1
        )
        return MixtureSameFamily(categorical_experts, ind_expert_dist)

    def log_prob(self, state, skill, next_state):
        n_next_state_delta = self.bb_out(next_state - state)
        prob = self.forward(state, skill)
        return prob.log_prob(n_next_state_delta)

    def loss(self, state, skill, next_state):
        log_prob = self.log_prob(state, skill, next_state)
        return -log_prob.mean()  # NLL

    def pred_next(self, deterministic=False):
        pass
