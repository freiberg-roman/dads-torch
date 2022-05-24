import hydra
from omegaconf import DictConfig, OmegaConf

from dads.env import create_env
from dads.main.dads_routines import eval_dads, train_dads, train_sac
from dads.planners import MPPIOptimizer


@hydra.main(config_path="configs", config_name="main.yaml")
def run(cfg: DictConfig):

    if cfg.mode == "mppi_test":
        env = create_env(cfg.env.name)
        mppi_planner = MPPIOptimizer(cfg.mppi)
        _, _, done, _ = env.reset()

        while not done:
            act = mppi_planner(env, replan=True)
            _, reward, done, _ = env.step(act)
            print("Current reward: ", reward)

    if cfg.mode == "train_sac":
        train_sac(cfg)

    if cfg.mode == "train_dads":
        train_dads(cfg)

    if cfg.mode == "eval_dads":
        eval_dads(cfg)

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run()
