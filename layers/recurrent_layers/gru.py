from copy import deepcopy
from typing import List, Optional

from models.adaptive_object import AdaptiveObject
from backend.backend import xp
from layers.activation_functions.sigmoid import Sigmoid
from layers.activation_functions.tanh import Tanh
from weight_initializers.random_initialize import rand_init


class GRU(AdaptiveObject):

    def __init__(self, in_units: int, hidden_units: int, name: str = "GRU layer", init_mode: str = "xavier_uniform"):
        super().__init__(name)
        self._init_mode = init_mode

        self._Vz = rand_init(hidden_units, in_units, init_mode)
        self._Vr = rand_init(hidden_units, in_units, init_mode)
        self._Vh = rand_init(hidden_units, in_units, init_mode)

        self._Wz = rand_init(hidden_units, hidden_units, init_mode)
        self._Wr = rand_init(hidden_units, hidden_units, init_mode)
        self._Wh = rand_init(hidden_units, hidden_units, init_mode)

        self._bz = xp.zeros((1, hidden_units), dtype=float)
        self._br = xp.zeros((1, hidden_units), dtype=float)
        self._bh = xp.zeros((1, hidden_units), dtype=float)

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        self._h: List[xp.ndarray] = list()  # čuvaćemo aktivnosti rekurentnih neurona ovog sloja za celu sekvencu
        self._hidden_units = hidden_units

        self._az: List[xp.ndarray] = [None]
        self._ar: List[xp.ndarray] = [None]
        self._ah: List[xp.ndarray] = [None]

        self.reset_state = True

        self._dEVz: Optional[xp.ndarray] = None
        self._dEVr: Optional[xp.ndarray] = None
        self._dEVh: Optional[xp.ndarray] = None

        self._dEWz: Optional[xp.ndarray] = None
        self._dEWr: Optional[xp.ndarray] = None
        self._dEWh: Optional[xp.ndarray] = None

        self._dEbz: Optional[xp.ndarray] = None
        self._dEbr: Optional[xp.ndarray] = None
        self._dEbh: Optional[xp.ndarray] = None

    def _pre_fw(self, inputs: xp.ndarray):
        super()._pre_fw(inputs)
        nb = inputs.shape[0]
        if self.reset_state or len(self._h) == 0 or self._h[0].shape[0] != nb:
            self._h = [xp.zeros((nb, self._hidden_units))]

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        """Pretpostavka kod prolaska unapred je da je new_input trodimenzionalni niz formata Nb x T x D,
            gde je Nb broj primera unutar batch-a, T dužina sekvence, a D veličina ulaza.

            Prolazak unapred implementiran je tako da propagiramo signal korak po korak. Tj. krećemo se po drugoj dimenziji
            tenzora new_input i generišemo izlaze i stanje korak po korak.
        """

        Nb = inputs.shape[0]
        T = inputs.shape[1]
        Y = xp.zeros((Nb, T, self._hidden_units))
        h_prev = self._h[0]

        for k in range(1, T + 1):
            # obratiti pažnju na indekse u nizu!

            u_k = inputs[:, k - 1, :]  # trenutni ulaz
            # U narednoj liniji računamo aktivacioni potencijal kapije ulaza (input gate)
            az_k = h_prev @ self._Wz.T + u_k @ self._Vz.T + self._bz
            ar_k = h_prev @ self._Wr.T + u_k @ self._Vr.T + self._bh

            r = self.sigmoid(ar_k)
            z = self.sigmoid(az_k)

            ah_k = (h_prev * r) @ self._Wh.T + u_k @ self._Vh.T + self._br

            h_new = (1 - z) * h_prev + z * self.tanh(ah_k)

            Y[:, k-1, :] = h_new

            if self.training:
                self._az.append(az_k)
                self._ar.append(ah_k)
                self._ah.append(ar_k)
                self._h.append(h_new)

            h_prev = h_new

        if not self.reset_state:
            self._h[0] = deepcopy(h_new)

        return Y

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        """dEdO je matrica parcijalnih izvoda greske po poslednjem izlazu.
            Kao i kod prolaska unapred, smatraćemo da je dEdO 3D tenzor formata  Nb x T x D.

            Implementacija u potpunosti prati pseudokod i formule date u prezentaciji sa predavanja
        """
        dEdU = xp.zeros_like(self._inputs)
        T = dEdO.shape[1]

        self._dEVz = xp.zeros_like(self._Vz)
        self._dEVh = xp.zeros_like(self._Vh)
        self._dEVr = xp.zeros_like(self._Vr)

        self._dEWz = xp.zeros_like(self._Wz)
        self._dEWh = xp.zeros_like(self._Wh)
        self._dEWr = xp.zeros_like(self._Wr)

        self._dEbz = xp.zeros_like(self._bz)
        self._dEbr = xp.zeros_like(self._br)
        self._dEbh = xp.zeros_like(self._bh)

        delta_h = dEdO[:, T - 1, :]

        for k in reversed(range(1, T+1)):
            # Da bi kod bio čitljiviji, izvući ćemo sve potrebne nizove u lokalne promenljive
            az_k = self._az[k]
            ar_k = self._ar[k]
            ah_k = self._ah[k]

            z_k = self.sigmoid(az_k)
            r_k = self.sigmoid(ar_k)
            htilda_k = self.tanh(ah_k)

            h_prev = self._h[k-1]
            uk = self._inputs[:, k-1, :]

            dz = self.sigmoid.deriv(az_k)
            dr = self.sigmoid.deriv(ar_k)

            dEdAz = delta_h * (htilda_k - h_prev) * dz
            dEdAh = delta_h * z_k * self.tanh.deriv(ah_k)
            dEdAr = (dEdAh @ self._Wh) * h_prev * dr

            # U narednim linijama računamo parcijalne izvode greške po matricama težina u koraku t do T
            # naravno, za celokupni parcijalni izvod potrebno je akumulirati parcijalne izvode

            self._dEWz += xp.matmul(dEdAz.T, h_prev)
            self._dEVz += xp.matmul(dEdAz.T, uk)
            self._dEbz += xp.sum(dEdAz, axis=0)

            #################################################

            self._dEWh += xp.matmul(dEdAh.T, h_prev)
            self._dEVh += xp.matmul(dEdAh.T, uk)
            self._dEbr += xp.sum(dEdAh, axis=0)

            #################################################

            self._dEWr += xp.matmul(dEdAr.T, h_prev)
            self._dEVr += xp.matmul(dEdAr.T, uk)
            self._dEbh += xp.sum(dEdAr, axis=0)

            #################################################

            dEdU[:, k-1, :] = xp.matmul(dEdAz, self._Vz) + \
                              xp.matmul(dEdAh, self._Vh) + \
                              xp.matmul(dEdAr, self._Vr)

            dEdHk = dEdAz @ self._Wz + (dEdAh @ self._Wh) * r_k + dEdAr @ self._Wr + delta_h * (1 - z_k)
            delta_h = dEdHk + dEdO[:, k - 1, :]

            self._az.pop(-1)
            self._ar.pop(-1)
            self._ah.pop(-1)
            self._h.pop(-1)

        return dEdU

    def update_parameters(self):
        self._optimizer.update_parameters(self._Vz, self._dEVz)
        self._optimizer.update_parameters(self._Vr, self._dEVr)
        self._optimizer.update_parameters(self._Vh, self._dEVh)

        self._optimizer.update_parameters(self._Wz, self._dEWz)
        self._optimizer.update_parameters(self._Wr, self._dEWr)
        self._optimizer.update_parameters(self._Wh, self._dEWh)

        self._optimizer.update_parameters(self._bz, self._dEbz)
        self._optimizer.update_parameters(self._br, self._dEbr)
        self._optimizer.update_parameters(self._bh, self._dEbh)

        self._dEVz = None
        self._dEVh = None
        self._dEVr = None

        self._dEWz = None
        self._dEWh = None
        self._dEWr = None

        self._dEbz = None
        self._dEbr = None
        self._dEbh = None

    @property
    def parameters(self) -> tuple:
        return self._Wz, self._Wh, self._Wr, \
               self._Vz, self._Vh, self._Vr, \
               self._bz, self._br, self._bh

    @parameters.setter
    def parameters(self, val: tuple):
        self._Wz, self._Wh, self._Wr, \
        self._Vz, self._Vh, self._Vr, \
        self._bz, self._br, self._bh = val
