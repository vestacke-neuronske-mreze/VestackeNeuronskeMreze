from backend.backend import xp
from loss_functions.abstract_loss_function import LossFunction


class MSE(LossFunction):
    def __init__(self):
        super().__init__('MeanSquareError')

    def backward(self, y: xp.ndarray, t: xp.ndarray) -> xp.ndarray:
        return (y - t) / y.shape[0]

    def __call__(self, y: xp.ndarray, t: xp.ndarray) -> float:
        return 0.5 * xp.mean(xp.square(y - t))
