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
        dims = [self.input_length, 64, 64, num_actions]
        self.net = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        # assert x.shape == (_, self.input_length), f'x.shape: {x.shape}'
        return self.net(x)