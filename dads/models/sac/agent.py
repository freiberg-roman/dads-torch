import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from dads.models.sac import GaussianPolicy, QNetwork
from dads.utils import EnvSteps
from dads.utils.math_helper import hard_update, soft_update


class SAC:
    def __init__(self, cfg, prep_state_fn=lambda x: x):

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha

        self.target_update_interval = cfg.target_update_interval
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning

        self.prep_state_fn = prep_state_fn
        self.device = torch.device("cuda" if cfg.device == "cuda" else "cpu")
        self.use_state_and_skill = cfg.env.skill_dim != 0

        self.critic = QNetwork(
            cfg.env.state_dim + cfg.env.skill_dim - cfg.env.num_coordinates,
            cfg.env.action_dim,
            cfg.hidden_size,
        ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr)

        self.critic_target = QNetwork(
            cfg.env.state_dim + cfg.env.skill_dim - cfg.env.num_coordinates,
            cfg.env.action_dim,
            cfg.hidden_size,
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(cfg.env.action_dim).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)

        self.policy = GaussianPolicy(
            cfg.env.state_dim + cfg.env.skill_dim - cfg.env.num_coordinates,
            cfg.env.action_dim,
            cfg.hidden_size,
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=cfg.lr)

    def select_action(self, state, skill=None, evaluate=False):
        state = self.prep_state_fn(state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.use_state_and_skill:
            skill = torch.FloatTensor(skill).to(self.device).unsqueeze(0)
            state = torch.cat((state, skill), dim=1)

        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch: EnvSteps):
        # Sample a batch from memory
        states, next_states, actions, rewards, dones, skills = batch.to_torch_batch()
        states = self.prep_state_fn(states)
        next_states = self.prep_state_fn(next_states)
        if self.use_state_and_skill:
            states = torch.cat((states, skills), dim=1)
            next_states = torch.cat((next_states, skills), dim=1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = rewards + (1 - dones.to(torch.float32)) * self.gamma * (
                min_qf_next_target
            )
        qf1, qf2 = self.critic(
            states, actions
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(states)

        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    # Save model parameters
    def save(self, base_path, folder):
        path = base_path + folder + "/sac/"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            path + "model.pt",
        )

    # Load model parameters
    def load(self, path, evaluate=False):
        ckpt_path = path + "/sac/model.pt"
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
