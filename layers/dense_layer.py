from models.adaptive_object import AdaptiveObject
from backend.backend import xp
from weight_initializers.random_initialize import rand_init


class DenseLayer(AdaptiveObject):
    """Izlaz gusto povezanog sloja dat je sa X*W^T + b^T.
       X je matrica dimenzija Nb x n_lPrev, W matrica težina dimenzija n_l x n_lPrev, a b vektor dimenzija n_l

       Sloj sadrži parametre koji se tokom učenja modifikuju pa je roditeljska klasa ovog sloja AdaptiveObject.
    """

    def __init__(self, input_units: int, output_units: int,
                 weight_init_method: str = 'xavier_uniform', name: str = 'unnamed'):
        super().__init__(name)

        self._W = rand_init(output_units, input_units, init_mode=weight_init_method)
        self._b = xp.zeros((output_units, ), dtype=float)
        self._dEdW = None
        self._dEdb = None

    @property
    def parameters(self) -> tuple:
        return self._W, self._b

    @parameters.setter
    def parameters(self, val: tuple):
        self._W, self._b = val

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        return xp.matmul(inputs, self._W.T) + self._b

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        einsum_expression = 'ki,kj->ij'  # ima isti efekat kao A.T x B

        if self._inputs.ndim == 3:
            einsum_expression = 'kti,ktj->ij'

        self._dEdW = xp.einsum(einsum_expression, dEdO, self._inputs)
        self._dEdb = xp.sum(dEdO, axis=tuple(range(dEdO.ndim - 1)))

        return xp.matmul(dEdO, self._W)

    def update_parameters(self):
        self._optimizer.update_parameters(self._W, self._dEdW)
        self._optimizer.update_parameters(self._b, self._dEdb)
        self._dEdW = None
        self._dEdb = None
