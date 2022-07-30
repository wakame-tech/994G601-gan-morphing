from functools import reduce
from typing import Tuple
import torch
import torch.nn as nn


def prod(t: tuple) -> int:
    return reduce(lambda x, y: x * y, t)

class Model(nn.Module):
    def __init__(self, input_shape: Tuple[int], num_actions: int):
        super(Model, self).__init__()
        self.input_length = prod(input_shape)
        self.fc1 = nn.Linear(self.input_length, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        x:
        """
        # assert x.shape == (1, self.input_length), f'x.shape: {x.shape}'
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.fc3(h)
        return h