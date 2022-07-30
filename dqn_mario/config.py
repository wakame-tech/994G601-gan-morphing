from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    device: str = 'cpu'
    batch_size: int = 128
    replay_momory_capacity: int = 100000
    target_update_frequency: int = 10
    n_episodes: int = 100
    n_steps: int = 100
    discount_factor: float = 0.98

    env_id: str = 'SuperMarioBros-v0'
    env_render: bool = True

    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 500

    actions: List[List[str]] = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
    ]