import math


class ConstantEps():
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self) -> float:
        return self.eps

class DecayEps():
    def __init__(self, begin: float, end: float, decay_steps: int) -> None:
        self.decay_steps = decay_steps
        self.steps = 0
        self.begin = begin
        self.end = end

    def reset(self):
        self.steps = 0

    def __call__(self) -> float:
        prog = math.exp(-self.steps / self.decay_steps)
        eps = self.end + (self.begin - self.end) * math.exp(prog)
        self.steps += 1
        return eps