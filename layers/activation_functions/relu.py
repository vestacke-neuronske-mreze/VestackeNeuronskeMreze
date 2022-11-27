from backend.backend import xp

from layers.activation_functions.activation_function import ActivationFunction


class ReLU(ActivationFunction):
    """ReLU funkcija definiše se kao f(x) = max(x, 0)
       Dakle, ponaša se kao identička za x > 0 i kao konstata 0 za x <= 0.
       Izvod identičke funkcije je 1, a konstante 0
    """

    def __init__(self, name="ReLU"):
        super().__init__(name=name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        return xp.maximum(inputs, 0)

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        df = xp.zeros_like(x)
        # df[input_tensor <= 0] = 0  # ne mora, jer vec inicijalizujemo nulama
        df[x > 0] = 1
        return df
