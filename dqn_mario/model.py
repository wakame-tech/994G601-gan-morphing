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
        self.net = nn.Sequential(
            nn.Linear(self.input_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor):
        # assert x.shape == (1, self.input_length), f'x.shape: {x.shape}'
        return self.net(x)