from pathlib import Path
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from eps_controller import ConstantEps, DecayEps

class Config:
    device: str
    env_id: str
    project_id: str
    model_dir: Path
    model_save_interval_episode: int

    seed: int = 0

    n_episodes: int
    start_episode: int
    n_steps: int

    env_render: bool
    reward_render: bool

    gamma: float
    # -- epsilon-greedy --
    eps: object

    # -- experience replay --
    replay_momory_capacity: int
    replay_memory_batch_size: int
    # -- fixed target network --
    target_update_frequency: int

class CartPoleConfig(Config):
    device: str = 'cpu'
    env_id: str = 'CartPole-v1'
    # actions = SIMPLE_MOVEMENT
    project_id: str = 'cartpole-v1'
    model_dir: Path = Path('models/')
    model_save_interval_episode: int = 500

    seed: int = 0

    n_episodes: int = 1000
    start_episode: int = 0
    n_steps: int = 500

    env_render: bool = False
    reward_render: bool = False

    gamma = 0.99
    # -- epsilon-greedy --
    # eps = ConstantEps(0.0)
    eps = DecayEps(1.0, 0.01, 100)

    # -- experience replay --
    replay_momory_capacity: int = 100000
    replay_memory_batch_size: int = 100
    # -- fixed target network --
    target_update_frequency: int = 20