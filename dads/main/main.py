import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="confings", config_name="main.yaml")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run()
