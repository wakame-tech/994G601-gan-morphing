
from gym import Env
import pandas as pd
from config import Config
from memory import SMBReplayMemory
from model_utils import load_model, save_model
import torch
from gym_env import make_env
from model import Model
import numpy as np
import torch.nn as nn
from datetime import datetime
import warnings

warnings.simplefilter('ignore')

def loss_step(config: Config, model: Model, target_model: Model, memory: SMBReplayMemory, criterion) -> float:
    state, action, reward, next_state, done, weights = memory.sample(config.batch_size)

    qs = model(state)
    qs_next = target_model(next_state)

    q = qs.gather(1, action.unsqueeze(-1)).squeeze(-1)
    q_next = qs_next.max(1)[0]
    expected_q = reward + config.gamma * q_next * (1 - done)

    # loss = (q - expected_q.detach()).pow(2) * torch.Tensor(weights)
    loss = criterion(q, expected_q)
    loss = loss.mean()
    loss.backward()
    return loss.item()

def make_action(model: Model, env: Env, state: np.ndarray, eps: float) -> int:
    if np.random.random() < eps:
        return env.action_space.sample()

    model.eval()
    state = state.reshape(1, -1)
    q: torch.Tensor = model.forward(torch.Tensor(state))
    return int(q.max(1)[1].item())

def train(
    config: Config,
    env: Env,
    model: Model,
    target_model: Model,
    memory: SMBReplayMemory,
    optimizer: torch.optim.Optimizer,
):
    df = pd.DataFrame(columns=['episode', 'step', 'loss', 'reward', 'eps', 'action', 'x_pos'])

    now = datetime.now()
    criterion = nn.SmoothL1Loss()

    for episode in range(config.n_episodes):
        delta = datetime.now() - now
        print(f'ep: {episode} @ {delta.total_seconds()}s')
        now = datetime.now()

        state: np.ndarray = env.reset()  # type: ignore
        ep_reward: float = 0.0
        x_pos = 0

        # get eps
        # eps is 1.0 when collectiong data to memory
        if len(memory) < config.batch_size:
            eps = 1.0
        else:
            eps = config.eps()

        for step in range(config.n_steps):
            # print(f'step: {step}')
            if config.env_render:
                env.render()


            action = make_action(model, env, state, eps)

            # step and update memory
            next_state, reward, done, info = env.step(action)
            x_pos = info['x_pos']
            memory.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # sync target model
            if step % config.target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

            # compute loss
            if len(memory) >= config.batch_size:
                loss = loss_step(config, model, target_model, memory, criterion)
                optimizer.step()
            else:
                loss = -1

            df = pd.concat([
                df,
                pd.DataFrame({
                    'episode': episode,
                    'step': step,
                    'loss': loss,
                    'reward': reward,
                    'eps': eps,
                    'action': action,
                    'x_pos': info['x_pos'],
                }, index=['episode', 'step'])
            ])

            if step % 10 == 0:
                print(f'[{episode:3d}-{step:4d}/{config.n_steps}] reward: {reward:.2f} (total: {ep_reward:.2f}), loss: {loss:.4f}, action: {config.actions[action]}, x_pos: {info["x_pos"]}')

            if done:
                break
        if episode % 10 == 0:
            save_model(config, model, optimizer, episode)

    df.to_csv(config.model_dir / f'{config.project_id}_log.csv', index=False)

if __name__ == '__main__':
    for rep_momory_size in [1000, 10000, 100000]:
        config = Config()
        config.project_id = f'smb-1-4-dqn_mem={rep_momory_size}'
        env = make_env(config)
        model, target_model, optimizer = load_model(config, env)
        memory = SMBReplayMemory(config.replay_momory_capacity)
        train(config, env, model, target_model, memory, optimizer)