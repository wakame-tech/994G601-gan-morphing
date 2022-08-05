
from gym import Env
from config import CartPoleConfig
from memory import EpisodeSteps, ReplayMemory, Step
from model_utils import load_model, save_model
from csv_logger import CSVLogger
import torch
from gym_env import make_env
from model import Model
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import warnings

from util import send_discord

warnings.simplefilter('ignore')

def train_loss(
    config: CartPoleConfig,
    model: Model,
    target_model: Model,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    returns train loss
    """
    states, actions, rewards, next_states, dones = memory.sample(config.replay_memory_batch_size)

    # predict q_t
    qs = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    # predit q_t+1
    qs_next = torch.zeros(config.replay_memory_batch_size)
    qs_next = target_model(next_states).max(1)[0]

    # * (1 - dones) means if done, qs[*] = 0
    expected_qs = rewards + (qs_next * config.gamma) * (1 - dones)
    loss = (qs - expected_qs).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def make_action(model: Model, env: Env, state: np.ndarray, eps: float) -> int:
    """
    random action with epsilon-greedy
    """
    if np.random.random() < eps:
        return env.action_space.sample()

    with torch.inference_mode():
        state = state.reshape(1, -1)
        q: torch.Tensor = model.forward(torch.Tensor(state))
        return int(q.max(1)[1].item())

def train2(
    config: CartPoleConfig,
    env: Env,
    model: Model,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    for episode in range(config.start_episode, config.n_episodes):
        state: np.ndarray = env.reset()  # type: ignore
        memory = EpisodeSteps(config.gamma)
        eps = config.eps(episode)

        for step in range(config.n_steps):
            # print(f'step: {step}')
            if config.env_render:
                env.render()

            x = Variable(torch.Tensor([state]))
            probs = torch.softmax(model(x), dim=1)

            action = np.random.choice(range(env.action_space.n), p=probs.detach().numpy()[0])
            next_state, reward, done, _ = env.step(action)
            # print(f'action: {action}, prob: {probs[0][action]}')
            memory.append(Step(probs[0][action], reward))
            state = next_state

            if done:
                break

        ep_reward, loss = memory.loss()
        print(f'episode: {episode}, loss: {loss:.2f}, ep_reward: {ep_reward}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logger.save()

def train(
    config: CartPoleConfig,
    env: Env,
    model: Model,
    target_model: Model,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
):
    """
    - Experience Replay
    - Fixed Target Network
    - Epsilon-Greedy
    """
    model.train()
    logger = CSVLogger(config, ['episode', 'eps', 'ep_reward', 'avg_loss'])
    now = datetime.now()

    for episode in range(config.start_episode, config.n_episodes):
        state: np.ndarray = env.reset()  # type: ignore
        ep_reward: float = 0.0
        avg_loss: float = 0.0

        eps = config.eps(episode)

        for step in range(config.n_steps):
            if config.env_render:
                env.render()

            action = make_action(model, env, state, eps)

            # step and update memory
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # sync target model
            if step % config.target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

            # compute loss
            loss = train_loss(config, model, target_model, memory, optimizer)

            avg_loss += loss

            # print(f'{step} step eps: {eps:.2f} reward: {reward:.2f} loss: {loss:.4f}')

            if done:
                delta = datetime.now() - now
                now = datetime.now()
                print(f'ep: {episode} @ {delta.total_seconds():.1f}s eps: {eps:.2f} total_reward: {ep_reward:.2f} loss: {loss:.4f}')
                break

        if episode % config.model_save_interval_episode == 0:
            message = f'[{config.project_id}/ep {episode:3d}] ep_reward: {ep_reward:.2f} loss: {loss:.4f}, eps: {eps:.2f}'
            send_discord(message)
            save_model(config, model, optimizer, memory, episode)


        logger.append({
            'episode': episode,
            'eps': eps,
            'ep_reward': ep_reward,
            'avg_loss': avg_loss / step,
        })

    logger.save()

def exp_cartpole():
    for cap in [
        10,
        1000,
        100000,
    ]:
        config = CartPoleConfig()
        env = make_env(config)
        config.replay_momory_capacity = cap
        config.project_id = f'{config.project_id}_mem={cap}'

        model, target_model, optimizer, memory = load_model(config, env)
        train(config, env, model, target_model, memory, optimizer)

    for batch_size in [
        10,
        50,
        100,
        500,
    ]:
        config = CartPoleConfig()
        env = make_env(config)
        config.replay_memory_batch_size = batch_size
        config.project_id = f'{config.project_id}_batch={batch_size}'

        model, target_model, optimizer, memory = load_model(config, env)
        train(config, env, model, target_model, memory, optimizer)


if __name__ == '__main__':
    exp_cartpole()