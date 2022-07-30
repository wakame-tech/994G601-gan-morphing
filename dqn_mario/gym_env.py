
from gym import Env
from gym.envs.registration import make
from gym.core import ObservationWrapper, Wrapper
from config import Config
import numpy as np

from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import cv2

class FrameDownsample(ObservationWrapper):
    def __init__(self, env, config: Config):
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

def score_fn(score, action, info) -> float:
    res = 0.0
    x_pos = info['x_pos']

    res += (x_pos - score) / 100.0
    if action == 6:
        res -= 3.0

    return res

class Reward(Wrapper):
    def __init__(self, env: Env):
        super(Reward, self).__init__(env)
        self.current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.current_score = score_fn(self.current_score, action, info)
        return state, reward, done, info

def make_env(config: Config) -> Env:
    from gym_super_mario_bros import make
    env = make(config.env_id)
    env = JoypadSpace(env, config.actions)
    env = FrameDownsample(env, config)
    env = Reward(env)
    return env