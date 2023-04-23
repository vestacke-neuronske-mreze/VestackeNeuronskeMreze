from backend.backend import xp
from layers.activation_functions.softmax import Softmax
from loss_functions.abstract_loss_function import LossFunction
from utils.utils import to_one_hot


class CrossEntropy(LossFunction):

    def __init__(self, from_logits: bool = True, one_hot: bool = True):
        super().__init__('CrosseEntropyLoss')
        self.from_logits = from_logits
        self.one_hot = one_hot

    def __call__(self, y: xp.ndarray, t: xp.ndarray) -> float:
        if not self.one_hot:
            t = to_one_hot(t, y.shape[-1])
        if self.from_logits:
            y = Softmax()(y)
        return -xp.sum(xp.log(y) * t) / t.shape[0]

    def backward(self, y: xp.ndarray, t: xp.ndarray) -> xp.ndarray:
        if not self.one_hot:
            t = to_one_hot(t, y.shape[-1])
        if self.from_logits:
            y = Softmax()(y)
            return (y - t) / t.shape[0]
        else:
            return -t/(y * y.shape[0])
