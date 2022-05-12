from torch.optim.adam import Adam

from dads.env import create_env
from dads.models import SAC, MixtureOfExperts
from dads.utils import EnvStep, RandomRB


def dads_train(cfg):

    env = create_env(cfg.overrides)
    buffer = RandomRB(cfg.buffer)
    agent = SAC(cfg.agent, env.prep_state())
    model = MixtureOfExperts(cfg.model, prep_input_fn=env.prep_state())
    model_optimizer = Adam(model.parameters(), lr=cfg.model.lr)

    if cfg.train.load_checkpoint != "":
        pass

    state, skill = env.reset()
    while env.total_steps < cfg.train.total_steps:
        # epoch
        for i in range(cfg.train.steps_per_epoch):
            # collect steps and compute intrinsic reward
            action = agent.select_action(state)
            next_state, reward, done, _, skill = env.step(action)
            reward = model.intrinsic_rew(
                EnvStep(state, next_state, action, 0.0, False, skill),
                env.sample_skills(cfg.train.reward_sampling_amount),
            )
            buffer.add(state, next_state, action, reward, done, skill)
            state = next_state

            if done:
                state, skill = env.reset()

            # update networks
            if i % cfg.train.update_model_every == 0:
                for batch in buffer.get_iter(
                    it=cfg.train.update_model_every, batch_size=cfg.train.batch_size
                ):
                    model.update_parameters(batch, model_optimizer)

            if i % cfg.train.update_agent_every == 0:
                for batch in buffer.get_iter(
                    it=cfg.train.update_agent_every, batch_size=cfg.train.batch_size
                ):
                    agent.update_parameters(batch)

        # end of epoch saving
