
from gym import Env, Wrapper


class Reward(Wrapper):
    def __init__(self, env: Env):
        super(Reward, self).__init__(env)
        self.x_pos = 0

    def score_fn(self, action, info) -> float:
        x_pos = info['x_pos']
        dx = -0.1 + (x_pos - self.x_pos) / 100.0
        goal_pt = 10.0 if info['flag_get'] else 0.0
        return dx + goal_pt

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.current_score = self.score_fn(action, info)
        self.x_pos = info['x_pos']
        return state, reward, done, info