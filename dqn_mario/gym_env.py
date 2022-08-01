
from gym import Env
# from gym.envs.registration import make
from gym_super_mario_bros import make
from gym.core import ObservationWrapper
from config import Config, SMBConfig
import numpy as np

from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import cv2

from reward import Reward

class FrameDownsample(ObservationWrapper):
    def __init__(self, env, config: SMBConfig):
        super(FrameDownsample, self).__init__(env)
        self.size = config.frame_size
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(*self.size, 1),
            dtype=np.uint8
        )

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame,
            self.size,
            interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


from gym_super_mario_bros import make

def make_env(config: Config) -> Env:
    env = make(config.env_id)
    # env = JoypadSpace(env, config.actions)
    # env = FrameDownsample(env, config)
    # env = Reward(env, config)

    print(f'state dim: {env.observation_space.shape} action dim: {env.action_space.n}')
    return env