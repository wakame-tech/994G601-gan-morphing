from collections import deque
from pathlib import Path
from tkinter import Variable
from typing import Tuple
import torch
import numpy as np
import pickle

class ReplayMemory:
    def __init__(self, capa: int):
        self.capa = capa
        self.buffer = deque()

    def save_pickle(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        push `env.step()` result
        if exceeds capacity, remove the oldest one
        """
        # state, next_state: (84, 84) -> (1, *)
        state_ = np.array(state.reshape(1, -1))
        next_state_ = np.array(next_state.reshape(1, -1))
        done_ = np.array(done)
        batch = (state_, action, reward, next_state_, done_)

        if len(self.buffer) > self.capa:
            self.buffer.popleft()

        self.buffer.append(batch)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sample from buffer

        return:
            states: (batch_size, state_dim)
            actions: (batch_size)
            rewards: (batch_size)
            next_states: (batch_size, state_dim)
            dones: (batch_size)
        """
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in indices]
        batch = tuple(zip(*samples))
        states, actions, rewards, next_states, dones = np.concatenate(batch[0]), batch[1], batch[2], np.concatenate(batch[3]), np.array(batch[4])
        states, actions, rewards, next_states, dones = torch.Tensor(states), torch.LongTensor(actions), torch.Tensor(rewards), torch.Tensor(next_states), torch.Tensor(dones)

        # uniformly weight
        # weights = np.array([1.0] * batch_size) / len(self.buffer)
        # weights = torch.Tensor(weights)
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)


def load_pickle(path: Path) -> ReplayMemory:
    with open(path, 'rb') as f:
        memory = pickle.load(f)
    return memory

from dataclasses import dataclass

@dataclass
class Step:
    confidence: Variable
    reward: float

class EpisodeSteps:
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.buffer = []

    def append(self, step: Step):
        self.buffer.append(step)

    def loss(self) -> Tuple[float, torch.Tensor]:
        loss = torch.tensor(1, dtype=torch.float32)
        for i, step in enumerate(self.buffer):
            reward = sum([(self.gamma ** j) * step.reward for j, step in enumerate(self.buffer[i:])])
            loss += -torch.log(step.confidence) * reward

        ep_reward = sum([(self.gamma ** j) * step.reward for j, step in enumerate(self.buffer)])
        return ep_reward, loss