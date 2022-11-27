from backend.backend import xp
from layers.activation_functions.activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self, name: str = "Tanh"):
        super().__init__(name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        ex = xp.exp(inputs)
        emx = xp.exp(-inputs)
        return (ex - emx) / (ex+emx)

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        tanh = self(x)
        return 1 - tanh*tanh
