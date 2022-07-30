from typing import List

from gym import Env, ObservationWrapper, make
import numpy as np
import cv2

from gym.spaces import Box
from nes_py.wrappers import JoypadSpace

class FrameDownsample(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = (84, 84)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(*self.size, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # (84, 84)
        frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


def make_env(env_id: str, action_space: List[List[str]]) -> Env:
    env = make(env_id)
    env = JoypadSpace(env, action_space)
    env = FrameDownsample(env)

    # todo

    return env