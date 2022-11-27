from backend.backend import xp

from layers.activation_functions.sigmoid import Sigmoid
from loss_functions.abstract_loss_function import LossFunction


class BinaryCrossEntropy(LossFunction):

    def __init__(self, from_logits: bool = False):
        super().__init__('BinaryCrosseEntropyLoss')
        self.from_logits = from_logits

    def __call__(self, y: xp.ndarray, t: xp.ndarray) -> float:
        if self.from_logits:
            y = Sigmoid()(y)

        # return -xp.nanmean(t * xp.log(y) + (1 - t) * xp.log(1 - y))
        return -xp.mean(t * xp.log(y) + (1 - t) * xp.log(1 - y))

    def backward(self, y: xp.ndarray, t: xp.ndarray) -> xp.ndarray:
        if self.from_logits:
            y = Sigmoid()(y)
            return (y - t) / y.shape[0]
        return (y - t) / (y * (1 - y) * y.shape[0])



