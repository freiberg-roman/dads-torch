from dads.env.mujoco.half_cheetah import HalfCheetahEnv
from dads.env.skill_env import SkillEnvironment


def create_env(env_name) -> SkillEnvironment:
    if env_name == "HalfCheetah":
        return SkillEnvironment(HalfCheetahEnv(), HalfCheetahEnv.get_env_conf())
