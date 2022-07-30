from pathlib import Path
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from eps_controller import ConstantEps

class Config:
    device: str = 'cpu'
    batch_size: int = 1
    replay_momory_capacity: int = 100000
    target_update_frequency: int = 30
    n_episodes: int = 30
    n_steps: int = 300
    discount_factor: float = 0.98

    # env_id: str = 'SuperMarioBros-v3'
    env_id: str = 'SuperMarioBros-v0'
    env_render: bool = True

    eps = ConstantEps(0.3)
    gamma = 0.99
    # frame_size = (84, 84)
    frame_size = (16, 9)

    # eps_start = 1.0
    # eps_end = 0.01
    # eps_decay = 500

    actions = SIMPLE_MOVEMENT
    project_id: str = 'smb-dqn'
    model_dir: Path = Path('models/')