from typing import Tuple
import torch
from gym import Env
from config import Config
from model import Model

def save_model(config: Config, model: Model, optimizer: torch.optim.Optimizer, episode: int):
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / f'{config.project_id}_{episode}.pth'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episode
    }, model_path)


def load_model(config: Config, env: Env) -> Tuple[Model, Model, torch.optim.Optimizer]:
    """
    returns train model and fixed target model
    """
    n = env.action_space.n
    space_shape = env.observation_space.shape

    model_path = config.model_dir / f'{config.project_id}_0.pth'

    model = Model(space_shape, n)
    optimizer = torch.optim.Adam(model.parameters())

    # if model_path.exists():
    #     state = torch.load(model_path)
    #     model.load_state_dict(state['model'])
    #     optimizer.load_state_dict(state['optimizer'])

    model.train()
    print(model)

    target_model = Model(space_shape, n)
    target_model.eval()

    return model, target_model, optimizer
