from omegaconf import OmegaConf

from dads.env import SkillEnvironment
from dads.models import SAC
from dads.env import create_env
from dads.utils import RandomRB


def test_simple_sac_routine():
    env_cfg = OmegaConf.create(
        {
            "name": "HalfCheetah",
            "state_dim": 18,
            "action_dim": 6,
            "num_coordinates": 1,
            "skill_dim": 2,
            "skill_continuous": True,
        }
    )

    hc_env: SkillEnvironment = create_env(env_cfg)
    buffer_cfg = OmegaConf.create({"capacity": 10000, "env": env_cfg})

    sac_cfg = OmegaConf.create(
        {
            "gamma": 0.99,
            "tau": 0.0005,
            "alpha": 0.05,
            "target_update_interval": 1,
            "automatic_entropy_tuning": True,
            "device": "cpu",
            "hidden_size": 1024,
            "lr": 0.0003,
            "env": env_cfg,
        }
    )

    sac_agent = SAC(sac_cfg, prep_state_fn=hc_env.prep_state())
    buffer = RandomRB(buffer_cfg)

    # Fill buffer
    s, skill = hc_env.reset()
    for _ in range(100):
        a = sac_agent.select_action(s, skill)
        s_n, r, d, i, skill = hc_env.step(a)
        buffer.add(s, s_n, a, r, d, skill)
        s = s_n

        if d:
            s, r, d, i, skill = hc_env.reset()

    for batch in buffer.get_iter(it=10, batch_size=100):
        sac_agent.update_parameters(batch)
