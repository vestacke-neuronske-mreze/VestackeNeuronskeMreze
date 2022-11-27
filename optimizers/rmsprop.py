from typing import Dict

from optimizers.abstract_optimizer import Optimizer
from backend.backend import xp, address


class RMSProp(Optimizer):
    """
    RMSProp optimizator vrši ažuriranje parametara dato sa:
    x_{t+1} = x_t - learning_rate * gradient_t / RMS(E[gradient]_t), gde je
    RMS(E[g]_t) = sqrt(E[sqr(gradient)]_t + eps), eps ~ 1e-8
    E[sqr(gradient)]_t = beta * E[sqr(gradient)]_{t-1} + (1 - beta) * gradient*gradient

    Akumulacija kvadrata gradijenta ima za cilj da tokom treniranja promena
    parametara bude veća za one parametre kod kojih do promena dolazi retko, a manja
    kod onih koji se često menjaju.
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.grad_sq: Dict[int, xp.ndarray] = {}
        self.eps = 1e-8

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        if address(params) not in self.grad_sq:
            self.grad_sq[address(params)] = xp.zeros_like(grad)

        grad_sq = self.grad_sq[address(params)]
        grad_sq = self.beta * grad_sq + (1 - self.beta) * grad * grad
        self.grad_sq[address(params)] = grad_sq
        params -= self.lr * grad / xp.sqrt(grad_sq + self.eps)

        return params
