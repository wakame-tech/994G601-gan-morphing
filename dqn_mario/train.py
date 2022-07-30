
from gym import Env
from config import Config
import torch
from game_io import make_env
from model import Model

def train(config: Config, env: Env):
    device = torch.device(config.device)
    def model(state, eps: float, device):
        return env.action_space.sample()

    # qp_net = Model()

    for episode in range(config.n_episodes):
        print(f'ep: {episode}')

        state = env.reset()
        ep_reward: float = 0.0

        for step in range(config.n_steps):
            # TODO: eps control
            eps = 1.0
            action = model(state, eps, device)

            if config.env_render:
                env.render()

            next_state, reward, done, info, *_ = env.step(action)
            state = next_state
            ep_reward += reward
            # print(f'reward: {reward} (total: {ep_reward})')

            # TODO: update model

            if done:
                # TODO:
                break

if __name__ == '__main__':
    config = Config()
    env = make_env(config.env_id, config.actions)
    train(config, env)