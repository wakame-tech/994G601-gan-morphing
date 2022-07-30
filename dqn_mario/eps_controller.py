class ConstantEps():
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self) -> float:
        return self.eps
