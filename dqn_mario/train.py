
from gym import Env
from monitor import Monitor
from torch import FloatTensor, Tensor
from config import Config
from memory import SMBReplayMemory
from model_utils import load_model, save_model
import torch
from gym_env import make_env
from model import Model
import numpy as np
from torch.autograd import Variable
from datetime import datetime

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

    if config.reward_render:
        plotter = Monitor(1, 100, 'steps', 'ep_reward')
    now = datetime.now()
    x_pos = 0
    for episode in range(config.n_episodes):
        if config.reward_render:
            plotter.reset()
        delta = datetime.now() - now
        print(f'ep: {episode} @ {delta.total_seconds()}s')
        now = datetime.now()

        state: np.ndarray = env.reset()  # type: ignore
        ep_reward: float = 0.0

        for step in range(config.n_steps):
            # print(f'step: {step}')
            if config.env_render:
                env.render()

            # eps controller
            eps = config.eps()
            action = make_action(model, env, state, eps)
            # step and update memory
            next_state, reward, done, info, *_ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            x_pos = info['x_pos']

            if done:
                # TODO:
                break

            # update model
            # sync
            if step % config.target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

            # compute loss
            optimizer.zero_grad()

            def loss_step(memory: SMBReplayMemory):
                state, action, reward, next_state, done, indices, weight = memory.sample(config.batch_size)
                state, action, reward, next_state, done = (
                    Variable(FloatTensor(np.float32(state))),
                    Variable(torch.LongTensor(action)),
                    Variable(torch.FloatTensor(reward)),
                    Variable(FloatTensor(np.float32(next_state))),
                    Variable(torch.FloatTensor(done)),
                )

                qs = model(state)
                qs_next = target_model(next_state)

                q = qs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                q_next = qs_next.max(1)[0]
                expected_q = reward + config.gamma * q_next * (1 - done)

                loss = (q - expected_q).pow(2) * Tensor(weight)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                return loss.item()

            loss = loss_step(memory)

            # print(f'ep: {episode:3d} step: {step:4d}/{config.n_steps} action: {action}, reward: {reward:2f} (total: {ep_reward:2f}), loss: {loss:.4f}')
            if config.reward_render:
                plotter.update(ep_reward)

        print(f'x_pos: {x_pos}')
        save_model(config, model, optimizer, episode)

if __name__ == '__main__':
    config = Config()
    env = make_env(config)
    model, target_model, optimizer = load_model(config, env)
    memory = SMBReplayMemory(config.replay_momory_capacity)

    train(config, env, model, target_model, memory, optimizer)