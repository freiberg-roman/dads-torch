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
        device="cuda",
    ):
        device = torch.device(device)
        self.planning_horizon = cfg.horizon
        self.population_size = cfg.population_size
        self.action_dimension = len(cfg.lower_bound)
        self.mean = torch.zeros(
            (self.planning_horizon, self.action_dimension),
            device=device,
            dtype=torch.float32,
        )

        self.lower_bound = torch.tensor(
            cfg.lower_bound, device=device, dtype=torch.float32
        )
        self.upper_bound = torch.tensor(
            cfg.upper_bound, device=device, dtype=torch.float32
        )
        self.var = cfg.sigma**2 * torch.ones_like(self.lower_bound)
        self.beta = cfg.beta
        self.gamma = cfg.gamma
        self.refinements = cfg.refinements

    def _eval_actions(self, actions, plannable):
        values = torch.zeros(len(actions))

        for i in len(actions):
            values[i] = plannable(actions[i])
        return values

    def set_model(self, model):
        pass

    def optimize(
        self,
        plannable,
    ) -> torch.Tensor:
        past_action = self.mean[0]
        self.mean[:-1] = self.mean[1:].clone()

        for _ in range(self.refinements):
            # sample noise and update constrained variances
            noise = torch.empty(
                size=(
                    self.population_size,
                    self.planning_horizon,
                    self.action_dimension,
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
            for i in range(max(self.planning_horizon - 1, 0)):
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
            values = self._eval_actions(population, plannable)
            values[values.isnan()] = -1e-10

            # weight actions
            weights = torch.reshape(
                torch.exp(self.gamma * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = population * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

        return self.mean.clone()
