
from typing import List
from gym import Env, Wrapper
from config import Config
from monitor import Monitor

class DxScore:
    def __init__(self) -> None:
        self.x_pos = 0
        self.pre_x_pos = 0

    def reset(self) -> None:
        self.x_pos = 0
        self.pre_x_pos = 0

    def update(self, action: int, info: dict):
        self.pre_x_pos = self.x_pos
        self.x_pos = info['x_pos']

    def score(self) -> float:
        return float(self.x_pos - self.pre_x_pos) * 3.0


class JumpScore:
    def __init__(self, actions: List[List[str]]) -> None:
        self.cont_jump_steps = 0
        self.cancel_jump = False
        self.actions = actions

    def reset(self) -> None:
        self.cancel_jump = False
        self.cont_jump_steps = 0

    def update(self, action: int, info: dict):
        # 'A' is jump button
        is_jump = 'A' in self.actions[action]
        if self.cont_jump_steps > 0 and not is_jump:
            self.cancel_jump = True
        else:
            self.cancel_jump = False

        if is_jump:
            self.cont_jump_steps += 1
        else:
            self.cont_jump_steps = 0

        # print('cont_jump_steps:', self.cont_jump_steps)

    def score(self) -> float:
        return max(0, float(self.cont_jump_steps - 3) * 10)


class GoalScore:
    def __init__(self):
        self.is_goal = False

    def reset(self) -> None:
        self.is_goal = False

    def update(self, action: int, info: dict):
        is_goal = info['flag_get']
        if is_goal:
            self.is_goal = True

    def score(self) -> float:
        if self.is_goal:
            return 10.0
        else:
            return 0.0


class NoopScore:
    def __init__(self) -> None:
        self.noop_steps = 0

    def reset(self):
        self.noop_steps = 0

    def update(self, action: int, info: dict):
        if action == 0:
            self.noop_steps += 1
        else:
            self.noop_steps = 0

    def score(self) -> float:
        return -float(self.noop_steps) * 0.1

class Reward(Wrapper):
    def __init__(self, env: Env, config: Config):
        super(Reward, self).__init__(env)
        self.config = config
        self.score_metrics = {
            'dx': DxScore(),
            'jump': JumpScore(config.actions),
            'goal': GoalScore(),
            'noop': NoopScore()
        }
        if config.reward_render:
            self.plotter = Monitor(1, 300, 'steps', 'reward')

    def reset(self):
        for metric in self.score_metrics.values():
            metric.reset()

        if self.config.reward_render:
            self.plotter.reset()

        return super().reset()

    def step(self, action: int):
        state, reward, done, info, *_ = self.env.step(action)

        # update & tally score
        scores = {}
        for name, score_metric in self.score_metrics.items():
            score_metric.update(action, info)
            score = score_metric.score()
            scores[name] = score

        # reward = reduce(lambda x, y: x + y, scores.values())

        if self.config.reward_render:
            self.plotter.update(reward)

        # print(f'scores: {scores}')
        return state, reward, done, info