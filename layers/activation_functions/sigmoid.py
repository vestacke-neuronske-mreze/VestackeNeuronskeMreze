from backend.backend import xp
from layers.activation_functions.activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    """s(x) = 1 / (1 + exp(-x))
        Lako može da se pokaže da je izvod sigmoidalne funkcije jednak:
        dsdx = s(x) * (1 - s(x))
    """

    def __init__(self, name: str = "Sigmoid"):
        super().__init__(name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        return 1 / (1 + xp.exp(-inputs))

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        y = self(x)
        return y * (1 - y)
