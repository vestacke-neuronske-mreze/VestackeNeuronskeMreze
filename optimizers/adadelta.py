from typing import Dict

from backend.backend import xp, address
from optimizers.abstract_optimizer import Optimizer


class Adadelta(Optimizer):
    """
    Adadelta algoritam rešava problem prestanka učenja koji je tipičan za Adagrad algoritam.
    Umesto računanja sume kvadrata svih gradijenata, Adadelta algoritam računa
    eksponencijalni pokretni prosek kvadrata gradijenata. Označimo sa E[g^2]t eksponencijalni
    pokretni prosek kvadrata gradijenata zaključno sa korakom t. Tada je
    E[g^2]t = βE[g^2]_{t-1} + (1 − β)g^2
    gde je β parametar koji kontroliše koliko uticaja prethodno izračunata prosečna vrednost
    ima u odnosu na novu vrednost gradijenta. Najčešće je β = 0.9. Neka je
    RMS[g]t = sqrt(E[g^2]t + eps)
    RMS[delta w]_{t−1} = sqrt(E[delta w ^2 ]_{t−1} + eps).

    Ažuriranje Adadelta algoritma dato je sledećim jednačinama:
    gt = ∇J(wt),
    E[g^2]t = βE[g^2]_{t-1} + (1 − β)g^2 ,
    delta wt = - RMS(delta w)_{t-1} * gt / RMS(g)_t
    E[delta w^2]t = βE[delta w^2]_{t−1} + (1 − β)delta w^2,
    w_{t+1} = wt + delta wt
    """

    def __init__(self, beta: float = 0.9):
        super().__init__(0)
        self.history: Dict[int, Dict[str, xp.ndarray]] = {}
        self.eps = 1e-6
        self.beta = beta

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        a = address(params)
        if a not in self.history:
            self.history[a] = {}
            self.history[a]["E_g_sq"] = xp.zeros_like(grad)
            self.history[a]["E_update_sq"] = xp.zeros_like(grad)

        self.history[a]["E_g_sq"] = self.beta * self.history[a]["E_g_sq"] + (1 - self.beta) * grad * grad
        update = - xp.sqrt(self.history[a]["E_update_sq"] + self.eps) * grad / xp.sqrt(self.history[a]["E_g_sq"] + self.eps)
        self.history[a]["E_update_sq"] = self.beta * self.history[a]["E_update_sq"] + (1 - self.beta) * update * update
        params += update

        return params
