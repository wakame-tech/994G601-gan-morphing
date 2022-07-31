from pathlib import Path
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from eps_controller import ConstantEps, DecayEps

class Config:
    device: str = 'cpu'
    env_id: str = 'SuperMarioBros-2-2-v0'
    actions = SIMPLE_MOVEMENT
    project_id: str = 'smb-dqn'
    model_dir: Path = Path('models/')
    batch_size: int = 100
    # frame_size = (84, 84)
    frame_size = (32, 18)

    env_render: bool = False
    reward_render: bool = False

    # -- about learning --
    replay_momory_capacity: int = 100000
    target_update_frequency: int = 30
    n_episodes: int = 100
    start_episode: int = 0
    n_steps: int = 1000
    # eps = ConstantEps(0.5)
    eps = DecayEps(1.0, 0.01, 500)
    gamma = 0.99