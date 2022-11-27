from copy import deepcopy
from typing import List, Optional

from models.adaptive_object import AdaptiveObject
from backend.backend import xp
from layers.activation_functions.sigmoid import Sigmoid
from layers.activation_functions.tanh import Tanh
from weight_initializers.random_initialize import rand_init


class LSTM(AdaptiveObject):

    def __init__(self, in_units: int, hidden_units: int, name: str = "LSTM layer", init_mode: str = "xavier_uniform"):
        super().__init__(name)
        self._init_mode = init_mode

        self._Vi = rand_init(hidden_units, in_units, init_mode)
        self._Vc = rand_init(hidden_units, in_units, init_mode)
        self._Vf = rand_init(hidden_units, in_units, init_mode)
        self._Vo = rand_init(hidden_units, in_units, init_mode)

        self._Wi = rand_init(hidden_units, hidden_units, init_mode)
        self._Wc = rand_init(hidden_units, hidden_units, init_mode)
        self._Wf = rand_init(hidden_units, hidden_units, init_mode)
        self._Wo = rand_init(hidden_units, hidden_units, init_mode)

        self._bi = xp.zeros((1, hidden_units), dtype=float)
        self._bf = xp.zeros((1, hidden_units), dtype=float)
        self._bc = xp.zeros((1, hidden_units), dtype=float)
        self._bo = xp.zeros((1, hidden_units), dtype=float)

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        self._h: List[xp.ndarray] = list()  # čuvaćemo aktivnosti rekurentnih neurona ovog sloja za celu sekvencu
        self._hidden_units = hidden_units

        self._ai: List[xp.ndarray] = [None]
        self._af: List[xp.ndarray] = [None]
        self._ac: List[xp.ndarray] = [None]
        self._ao: List[xp.ndarray] = [None]

        self._c: List[xp.ndarray] = [None]

        self.reset_state = True

        self._dEVi: Optional[xp.ndarray] = None
        self._dEVc: Optional[xp.ndarray] = None
        self._dEVf: Optional[xp.ndarray] = None
        self._dEVo: Optional[xp.ndarray] = None

        self._dEWi: Optional[xp.ndarray] = None
        self._dEWc: Optional[xp.ndarray] = None
        self._dEWf: Optional[xp.ndarray] = None
        self._dEWo: Optional[xp.ndarray] = None

        self._dEbi: Optional[xp.ndarray] = None
        self._dEbf: Optional[xp.ndarray] = None
        self._dEbc: Optional[xp.ndarray] = None
        self._dEbo: Optional[xp.ndarray] = None

    def _pre_fw(self, inputs: xp.ndarray):
        super()._pre_fw(inputs)
        nb = inputs.shape[0]

        if self.reset_state or len(self._h) == 0 or self._h[0].shape[0] != nb:
            self._h = [xp.zeros((nb, self._hidden_units))]
            self._c = [xp.zeros((nb, self._hidden_units))]

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
        c_prev = self._c[0]
        for k in range(1, T + 1):
            # obratiti pažnju na indekse u nizu!

            u_k = inputs[:, k - 1, :]  # trenutni ulaz
            # U narednoj liniji računamo aktivacioni potencijal kapije ulaza (input gate)
            ai_k = h_prev @ self._Wi.T + u_k @ self._Vi.T + self._bi
            af_k = h_prev @ self._Wf.T + u_k @ self._Vf.T + self._bf
            ac_k = h_prev @ self._Wc.T + u_k @ self._Vc.T + self._bc
            ao_k = h_prev @ self._Wo.T + u_k @ self._Vo.T + self._bo

            # Sada da na svaki potencijal primenimo odgovarajuću aktivacionu funkciju...
            i = self.sigmoid(ai_k)
            f = self.sigmoid(af_k)
            c_tilda = self.tanh(ac_k)
            o = self.sigmoid(ao_k)

            c_new = f * c_prev + i * c_tilda
            h_new = o * self.tanh(c_new)

            Y[:, k-1, :] = h_new

            if self.training:
                self._ai.append(ai_k)
                self._af.append(af_k)
                self._ac.append(ac_k)
                self._ao.append(ao_k)

                self._c.append(c_new)
                self._h.append(h_new)

            c_prev = c_new
            h_prev = h_new

        if not self.reset_state:
            # ovaj atribut je podrazumevano na False, pa se ova grana najčešće neće izvršavati.
            # ukoliko je na True, poslednje stanje rekurentnih neurona trenutne sekvence biće početno stanje za
            # narednu sekvencu. Time dobijamo efekat kao da radimo sa dužim sekvencama.
            # inače je _x[0] matrica sa nulama
            self._h[0] = deepcopy(h_new)
            self._c[0] = deepcopy(c_new)

        return Y

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        """dEdO je matrica parcijalnih izvoda greske po poslednjem izlazu.
            Kao i kod prolaska unapred, smatraćemo da je dEdO 3D tenzor formata  Nb x T x D.

            Implementacija u potpunosti prati pseudokod i formule date u prezentaciji sa predavanja
        """
        dEdU = xp.zeros_like(self._inputs)
        T = dEdO.shape[1]

        self._dEVi = xp.zeros_like(self._Vi)
        self._dEVf = xp.zeros_like(self._Vf)
        self._dEVc = xp.zeros_like(self._Vc)
        self._dEVo = xp.zeros_like(self._Vo)

        self._dEWi = xp.zeros_like(self._Wi)
        self._dEWf = xp.zeros_like(self._Wf)
        self._dEWc = xp.zeros_like(self._Wc)
        self._dEWo = xp.zeros_like(self._Wo)

        self._dEbi = xp.zeros_like(self._bi)
        self._dEbf = xp.zeros_like(self._bf)
        self._dEbc = xp.zeros_like(self._bc)
        self._dEbo = xp.zeros_like(self._bo)

        for k in reversed(range(1, T+1)):
            # Da bi kod bio čitljiviji, izvući ćemo sve potrebne nizove u lokalne promenljive
            ai_k = self._ai[k]
            af_k = self._af[k]
            ac_k = self._ac[k]
            ao_k = self._ao[k]

            i_k = self.sigmoid(ai_k)
            # f_t = self._f[k]
            c_tilde_k = self.tanh(ac_k)
            o_k = self.sigmoid(ao_k)
            c_k = self._c[k]
            c_prev = self._c[k-1]
            h_prev = self._h[k-1]
            uk = self._inputs[:, k-1, :]

            # U toku računanja između ostalog biće nam potrebni i izvodi svih aktivacionih funkcija,
            # pa ćemo ih pronaći na samom početku
            di = self.sigmoid.deriv(ai_k)
            df = self.sigmoid.deriv(af_k)
            # dcand_mem = self.tanh.deriv(ac_t)
            do = self.sigmoid.deriv(ao_k)
            dc = self.tanh.deriv(c_k)

            dEdc = dEdO[:, -1, :] * o_k * dc

            if k == T:
                delta_c = dEdc
                delta_h = dEdO[:, k-1, :]
            else:
                dEdHk = dEdAi @ self._Wi + dEdAf @ self._Wf + dEdAc @ self._Wc + dEdAo @ self._Wo
                delta_c = dEdHk * o_k * dc + dEdc
                delta_h = dEdHk + dEdO[:, k-1, :]

            dEdAi = delta_c * c_tilde_k * di
            dEdAf = delta_c * c_prev * df
            dEdAc = delta_c * i_k * dc
            dEdAo = delta_h * self.tanh(c_k) * do

            # U narednim linijama računamo parcijalne izvode greške po matricama težina u koraku t do T
            # naravno, za celokupni parcijalni izvod potrebno je akumulirati parcijalne izvode

            self._dEWi += xp.matmul(dEdAi.T, h_prev)
            self._dEVi += xp.matmul(dEdAi.T, uk)
            self._dEbi += xp.sum(dEdAi, axis=0)

            #################################################

            self._dEWf += xp.matmul(dEdAf.T, h_prev)
            self._dEVf += xp.matmul(dEdAf.T, uk)
            self._dEbf += xp.sum(dEdAf, axis=0)

            #################################################

            self._dEWc += xp.matmul(dEdAc.T, h_prev)
            self._dEVc += xp.matmul(dEdAc.T, uk)
            self._dEbc += xp.sum(dEdAc, axis=0)

            #################################################

            self._dEWo += xp.matmul(dEdAo.T, h_prev)
            self._dEVo += xp.matmul(dEdAo.T, uk)
            self._dEbo += xp.sum(dEdAo, axis=0)

            #################################################

            dEdU[:, k-1, :] = xp.matmul(dEdAi, self._Vi) + \
                                    xp.matmul(dEdAf, self._Vf) + \
                                    xp.matmul(dEdAc, self._Vc) + \
                                    xp.matmul(dEdAo, self._Vo)

            self._ai.pop(-1)
            self._af.pop(-1)
            self._ac.pop(-1)
            self._ao.pop(-1)
            self._c.pop(-1)
            self._h.pop(-1)

        return dEdU

    def update_parameters(self):
        self._optimizer.update_parameters(self._Vi, self._dEVi)
        self._optimizer.update_parameters(self._Vc, self._dEVc)
        self._optimizer.update_parameters(self._Vf, self._dEVf)
        self._optimizer.update_parameters(self._Vo, self._dEVo)

        self._optimizer.update_parameters(self._Wi, self._dEWi)
        self._optimizer.update_parameters(self._Wc, self._dEWc)
        self._optimizer.update_parameters(self._Wf, self._dEWf)
        self._optimizer.update_parameters(self._Wo, self._dEWo)

        self._optimizer.update_parameters(self._bi, self._dEbi)
        self._optimizer.update_parameters(self._bf, self._dEbf)
        self._optimizer.update_parameters(self._bc, self._dEbc)
        self._optimizer.update_parameters(self._bo, self._dEbo)

        self._dEVi = None
        self._dEVf = None
        self._dEVc = None
        self._dEVo = None

        self._dEWi = None
        self._dEWf = None
        self._dEWc = None
        self._dEWo = None

        self._dEbi = None
        self._dEbf = None
        self._dEbc = None
        self._dEbo = None

    @property
    def parameters(self) -> tuple:
        return self._Wi, self._Wf, self._Wc, self._Wo, \
               self._Vi, self._Vf, self._Vc, self._Vo, \
               self._bi, self._bf, self._bc, self._bo,

    @parameters.setter
    def parameters(self, val: tuple):
        self._Wi, self._Wf, self._Wc, self._Wo, \
        self._Vi, self._Vf, self._Vc, self._Vo, \
        self._bi, self._bf, self._bc, self._bo = val
