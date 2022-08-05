from typing import Tuple
from memory import ReplayMemory, load_pickle
import torch
from gym import Env
from config import Config
from model import Model

def save_model(config: Config, model: Model, optimizer: torch.optim.Optimizer, memory: ReplayMemory, episode: int):
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / f'{config.project_id}_{episode}.pth'
    memory_path = config.model_dir / f'{config.project_id}_{episode}_memory.pkl'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episode
    }, model_path)
    memory.save_pickle(memory_path)
    print(f'save model to {model_path}, {memory_path}')


def load_model(config: Config, env: Env) -> Tuple[Model, Model, torch.optim.Optimizer, ReplayMemory]:
    """
    returns train model and fixed target model
    """
    n: int = env.action_space.n  # type: ignore
    space_shape: tuple = env.observation_space.shape  # type: ignore

    model_path = config.model_dir / f'{config.project_id}_{config.start_episode}.pth'
    memory_path = config.model_dir / f'{config.project_id}_{config.start_episode}_memory.pkl'

    model = Model(space_shape, n)
    optimizer = torch.optim.Adam(model.parameters())

    memory = ReplayMemory(config.replay_momory_capacity)

    if False and model_path.exists() and memory_path.exists():
        memory = load_pickle(memory_path)
        state = torch.load(model_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f'load model from {model_path}, {memory_path}')

    print(model)

    target_model = Model(space_shape, n)

    return model, target_model, optimizer, memory
