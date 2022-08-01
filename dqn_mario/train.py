
from gym import Env
from config import CartPoleConfig, Config
from memory import ReplayMemory
from model_utils import load_model, save_model
from csv_logger import CSVLogger
import torch
from gym_env import make_env
from model import Model
import numpy as np
import torch.nn as nn
from datetime import datetime
import warnings
from icecream import ic

warnings.simplefilter('ignore')

def train_model(
    config: Config,
    model: Model,
    target_model: Model,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    criterion
) -> float:
    """
    returns train loss
    """
    model.train()
    states, actions, rewards, next_states, dones = memory.sample(config.replay_memory_batch_size)

    # predict q_t
    qs = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    # predit q_t+1
    qs_next = torch.zeros(config.replay_memory_batch_size)
    qs_next = target_model(next_states).max(1)[0].detach()

    # * (1 - dones) means if done, qs[*] = 0
    expected_qs = (qs_next * config.gamma) + rewards * (1 - dones)

    loss = criterion(qs, expected_qs)

    optimizer.zero_grad()
    loss.backward()
    loss = loss.mean()
    optimizer.step()

    return loss.item()

def make_action(model: Model, env: Env, state: np.ndarray, eps: float) -> int:
    """
    random action with epsilon-greedy
    """
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
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
):
    # logger = CSVLogger(config, ['episode', 'step', 'loss', 'reward', 'ep_reward', 'eps', 'action', 'x_pos'])
    logger = CSVLogger(config, ['episode', 'step', 'loss', 'reward', 'ep_reward', 'eps', 'action'])
    now = datetime.now()
    criterion = nn.SmoothL1Loss()

    for episode in range(config.start_episode, config.n_episodes):
        state: np.ndarray = env.reset()  # type: ignore
        ep_reward: float = 0.0

        # get eps
        # eps is 1.0 when collectiong data to memory
        if len(memory) < config.replay_memory_batch_size:
            eps = 1.0
        else:
            eps = config.eps(episode)

        for step in range(config.n_steps):
            # print(f'step: {step}')
            if config.env_render:
                env.render()


            action = make_action(model, env, state, eps)

            # step and update memory
            next_state, reward, done, *_ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # sync target model
            if step % config.target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

            # compute loss
            loss = train_model(config, model, target_model, memory, optimizer, criterion)

            logger.append({
                'episode': episode,
                'step': step,
                'loss': loss,
                'reward': reward,
                'ep_reward': ep_reward,
                'eps': eps,
                'action': action,
                # 'x_pos': info['x_pos'],
            })

            # print(f'[{episode:3d}-{step:4d}/{config.n_steps}] reward: {reward:.2f} (total: {ep_reward:.2f}), loss: {loss:.4f}, action: {config.actions[action]}, x_pos: {info["x_pos"]}')
            # print(f'[{episode:3d}-{step:4d}/{config.n_steps}] reward: {reward:.2f} (total: {ep_reward:.2f}), loss: {loss:.4f}, action: {config.actions[action]}')

            if done:
                delta = datetime.now() - now
                now = datetime.now()
                print(f'ep: {episode} @ {delta.total_seconds():.1f}s eps: {eps:.2f} total_reward: {ep_reward:.2f}')
                break

        if episode % config.model_save_interval_episode == 0:
            save_model(config, model, optimizer, memory, episode)

        logger.save()


def exp_mem(course: str):
    env_id = f'SuperMarioBros-{course}-v0'
    for rep_momory_size in [
        1000,
        10000,
        100000,
    ]:
        config = Config()

        # continue
        config.env_id = env_id
        config.project_id = f'{config.project_id}-{course}_mem={rep_momory_size}'

        env = make_env(config)
        model, target_model, optimizer, memory = load_model(config, env)
        train(config, env, model, target_model, memory, optimizer)


def exp_cartpole():
    config = CartPoleConfig()
    env = make_env(config)

    model, target_model, optimizer, memory = load_model(config, env)
    train(config, env, model, target_model, memory, optimizer)

if __name__ == '__main__':
    exp_cartpole()
    # exp_mem('1-1')
    # exp_mem('2-2')