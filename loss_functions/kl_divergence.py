from typing import Tuple

from backend.backend import xp
from loss_functions.abstract_loss_function import LossFunction


class DKLStandardNormal(LossFunction):
    """Specijalni slučaj KL divergencije gde računamo Dkl između raspodele q koja je Gausova sa parametrima mi_q i exp(gamma) i
       p koja je takođe Gausova sa parametrima mi_p = 0 i var_p = 1
    """
    def __init__(self):
        super().__init__('KL Divergence')

    def __call__(self, mu: xp.ndarray, gamma: xp.ndarray) -> xp.ndarray:
        return -0.5 * xp.sum(1 + gamma - mu ** 2 - xp.exp(gamma)) / gamma.shape[0]

    def backward(self, mu: xp.ndarray, gamma: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        """Ovde treba računati parcijalne izvode DKL po mi i gama.
        Moramo samo paziti gde tako izračunate parcijalne izvode smeštamo. Kod propagacije izvoda unazad
        oni će kao i kod ulaza biti spojeni u jedan, duži, niz. """
        nb = mu.shape[0]
        return mu / nb, 0.5 * (xp.exp(gamma) - 1) / nb
