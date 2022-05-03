from dads.env.mujoco.half_cheetah import HalfCheetahEnv


def create_env(env_name):
    if env_name == "HalfCheetah":
        return HalfCheetahEnv()
