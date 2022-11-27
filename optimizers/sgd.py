from backend.backend import xp

from optimizers.abstract_optimizer import Optimizer


class SGD(Optimizer):
    """Osnovni gradijentni spust je najjednostavniji algoritam optimizacije.
       Parametri se ažuriraju po formuli:
       x = x - lr * g, gde je lr korak učenja, a g gradijent.
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def update_parameters(self, params: xp.ndarray,
                          grad: xp.ndarray) -> xp.ndarray:
        params -= self.lr * grad

        return params
