from collections import deque
from typing import Tuple
import torch
import numpy as np

class SMBReplayMemory:
    def __init__(self, capa: int):
        self.capa = capa
        self.buffer = deque()

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

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        sample from buffer
        """
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in indices]
        batch = tuple(zip(*samples))
        states, actions, rewards, next_states, dones = np.concatenate(batch[0]), batch[1], batch[2], np.concatenate(batch[3]), np.array(batch[4])
        states, actions, rewards, next_states, dones = torch.Tensor(states), torch.LongTensor(actions), torch.Tensor(rewards), torch.Tensor(next_states), torch.Tensor(dones)

        # uniformly weight
        weights = np.array([1.0] * batch_size) / len(self.buffer)
        return states, actions, rewards, next_states, dones, weights


    def __len__(self):
        return len(self.buffer)