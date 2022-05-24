import torch

from dads.utils.math_helper import truncated_normal


class MPPIOptimizer:
    """Implements the Model Predictive Path Integral optimization algorithm.
    A derivation of MPPI can be found at https://arxiv.org/abs/2102.09027
    This version is closely related to the original TF implementation used in PDDM with
    some noise sampling modifications and the addition of refinement steps.
    """

    def __init__(
        self,
        cfg,
        agent,
        plannable,
        env_reward,
        device="cpu",
    ):
        self.device = torch.device(device)
        device = torch.device(device)
        self.horizon_planning = cfg.horizon_planning
        self.horizon_z = cfg.horizon_z
        self.population_size = cfg.population_size
        self.skill_dim = cfg.env.skill_dim
        self.mean = torch.zeros(
            (self.horizon_planning, self.skill_dim),
            device=device,
            dtype=torch.float32,
        )

        self.lower_bound = torch.tensor(
            cfg.env.skill_lower_bound, device=device, dtype=torch.float32
        )
        self.upper_bound = torch.tensor(
            cfg.env.skill_upper_bound, device=device, dtype=torch.float32
        )
        self.var = cfg.sigma**2 * torch.ones_like(self.lower_bound)
        self.beta = cfg.beta
        self.gamma = cfg.gamma
        self.refinements = cfg.refinements
        self.agent = agent
        self.env_reward = env_reward
        self.plannable = plannable

    def _eval(self, init_state, skills):
        state = torch.zeros(
            (self.population_size, len(init_state)),
            dtype=torch.float32,
            device=self.device,
        )
        state[:, :] = torch.tensor(init_state)
        values = torch.zeros((self.population_size,))

        for i in range(self.horizon_planning):
            next_state = torch.squeeze(self.plannable.pred_next(state, skills[:, i, :]))
            values += self.env_reward(state, next_state)
            state = next_state
        return values

    def set_model(self, model):
        pass

    def __call__(self, init_state) -> torch.Tensor:
        past_action = self.mean[0]
        self.mean[:-1] = self.mean[1:].clone()

        for _ in range(self.refinements):
            # sample noise and update constrained variances
            noise = torch.empty(
                size=(
                    self.population_size,
                    self.horizon_planning,
                    self.skill_dim,
                ),
                device=self.device,
            )
            noise = truncated_normal(noise)

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.minimum(mv, self.var)
            population = noise.clone() * torch.sqrt(constrained_var)

            # smoothed actions with noise
            population[:, 0, :] = (
                self.beta * (self.mean[0, :] + noise[:, 0, :])
                + (1 - self.beta) * past_action
            )
            for i in range(max(self.horizon_planning - 1, 0)):
                population[:, i + 1, :] = (
                    self.beta * (self.mean[i + 1] + noise[:, i + 1, :])
                    + (1 - self.beta) * population[:, i, :]
                )
            # clipping actions
            # This should still work if the bounds between dimensions are different.
            population = torch.where(
                population > self.upper_bound, self.upper_bound, population
            )
            population = torch.where(
                population < self.lower_bound, self.lower_bound, population
            )
            values = self._eval(init_state, population)
            values[values.isnan()] = -1e-10

            # weight actions
            weights = torch.reshape(
                torch.exp(self.gamma * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = population * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

        return self.mean.clone()[0, :]
