from typing import Union, Dict

from optimizers.abstract_optimizer import Optimizer
from backend.backend import xp, address


class AdaMax(Optimizer):
    """
    Slično Adadelta i RMSprop algoritmu, i Adam (engl. Adaptive Moment Estimation)
    koristi eksponencijalne pokretne proseke gradijenata i kvadrata gradijenata. U koraku t
    definišemo vektore mt i vt kao:
    mt = β1 mt + (1 − β1 )gt,
    vt = β2vt + (1 − β2 )gt^2
    Vektori m i v bi trebalo da estimiraju gradijent i kvadrat gradijenta. Međutim, ažuriranja
    data navedenim jednačinama dovode do pristrasne procene, pa treba izvršiti sledeću korekciju:
    mt_corr = mt/(1 − β1^t),
    vt_corr = vt /(1 − β2^t).
    Sa tako izračunatim parametrima mt_corr i vt_corr, parametre ažuriramo koristeći jednačinu
    w_{t+1} = wt − α mt_corr/( sqrt(vt_corr) + eps)
    gde je eps konstanta reda veličine 10−8, a parametar β2 najčešće iznosi 0.999
    """

    def __init__(self, lr: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.history: Dict[int, Dict[str, Union[xp.ndarray, int]]] = {}
        self.eps = 1e-8

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        a = address(params)
        if a not in self.history:
            self.history[a] = {}
            self.history[a]["t"] = 0
            self.history[a]["v"] = xp.zeros_like(grad)
            self.history[a]["m"] = xp.zeros_like(grad)

        self.history[a]["t"] += 1
        self.history[a]["m"] = self.beta_1 * self.history[a]["m"] + (1 - self.beta_1) * grad
        self.history[a]["v"] = xp.maximum(self.beta_2 * self.history[a]["v"], xp.abs(grad))

        m_corr = self.history[a]["m"] / (1 - xp.power(self.beta_1, self.history[a]["t"]))

        params -= self.lr * m_corr / (self.history[a]["v"] + self.eps)

        return params
