from dads.env.mujoco.half_cheetah import HalfCheetahEnv
from dads.env.skill_env import SkillEnvironment


def create_env(cfg, record=False) -> SkillEnvironment:
    if cfg.name == "HalfCheetah":
        return SkillEnvironment(HalfCheetahEnv(), cfg, record=record)
