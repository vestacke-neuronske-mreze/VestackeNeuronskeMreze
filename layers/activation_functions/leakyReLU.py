from backend.backend import xp
from layers.activation_functions.activation_function import ActivationFunction


class LeakyReLU(ActivationFunction):

    def __init__(self, name="LeakyReLU", alpha: float = 0.3):
        super().__init__(name=name)
        self.alpha = alpha

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        y = xp.zeros_like(inputs)
        y[inputs >= 0] = inputs[inputs >= 0]
        y[inputs < 0] = inputs[inputs < 0] * self.alpha
        return y

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        df = xp.zeros_like(x)
        df[x >= 0] = 1
        df[x < 0] = self.alpha
        return df
