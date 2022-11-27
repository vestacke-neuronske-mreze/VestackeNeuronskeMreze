from typing import Dict

from backend.backend import xp, address
from optimizers.abstract_optimizer import Optimizer


class Adagrad(Optimizer):
    """
    Osnovna ideja Adagrad algoritma sastoji se u tome da se tokom treniranja promena
    parametara bude veća za one parametre kod kojih do promena dolazi retko, a manja
    kod onih koji se često menjaju. To se postiže uvođenjem dijagonalne matrice G ∈ Rd×d.
    Označimo sa G(t) matricu u koraku t. Ažuriranje Adagrad algoritma dato je sledećim
    jednačinama
    gt = ∇J(wt),
    Gi,i (t) = Gi,i(t−1) + (gt,i)^2, ∀i ∈ {1, . . . , d},
    wt+1,i = wt,i − αgt,i / sqrt(Gi,i + eps), ∀i ∈ {1, . . . , d}.

    Matrica G akumulira kvadrate gradijenata i na osnovu njih povećava ili smanjuje uti-
    caj koji određena komponenta gradijenta ima na parametre. Parametar eps je konstanta
    dodata radi numeričke stabilnosti i obično iznosi eps = 10−8. Što je veća vrednost odred̄enog
    elementa Gi,i, to će manji uticaj komponente gi biti na parametar wi , a vrednost dija-
    gonalnog elementa matrice G biće veća za one parametre kod kojih su promene česte.
    Međutim, vremenom će elementi matrice G povećati u meri da će učenje biti onemo-
    gućeno, što je mana ovog algoritma.

    Naravno, umesto dijagonalne matrice G dovoljno je da koristimo ndimenzionalni niz istih
    dimenzija kao i parametri, a sve operacije mogu biti pokomponentne.
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)
        self.g_sq: Dict[int, xp.ndarray] = {}
        self.eps = 1e-8

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        a = address(params)

        if a not in self.g_sq:
            self.g_sq[a] = xp.zeros_like(grad)

        g_sq = self.g_sq[a]
        g_sq += grad*grad
        self.g_sq[a] = g_sq
        params -= self.lr * grad / xp.sqrt(g_sq + self.eps)

        return params
