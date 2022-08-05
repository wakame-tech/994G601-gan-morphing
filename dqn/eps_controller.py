import math


class ConstantEps():
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, epsode: int) -> float:
        return self.eps

class DecayEps():
    """
    decayed eplison-greedy policy
    """
    def __init__(self, begin: float, end: float, decay_episodes: int) -> None:
        assert begin > end, f'begin: {begin}, end: {end}'
        self.decay_episodes = decay_episodes
        self.begin = begin
        self.end = end

    def __call__(self, episode: int) -> float:
        r = math.exp(-1. * episode / self.decay_episodes)
        # correct
        return self.end + (self.begin - self.end) * r