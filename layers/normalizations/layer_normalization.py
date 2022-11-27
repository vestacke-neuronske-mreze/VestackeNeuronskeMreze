from models.adaptive_object import AdaptiveObject
from backend.backend import xp


class LayerNormalization(AdaptiveObject):

    def __init__(self, name: str = 'Layer normalization layer'):
        super().__init__(name)

        self.mean = None
        self.var = None
        self.gamma = None
        self.d_gamma = None
        self.beta = None
        self.d_beta = None
        self._tmp_array = None
        self.x_hat = None

    @property
    def parameters(self) -> tuple:
        return self.gamma, self.beta

    @parameters.setter
    def parameters(self, val: tuple):
        self.gamma, self.beta = val

    def _pre_fw(self, inputs: xp.ndarray):
        super()._pre_fw(inputs)
        if self.beta is None:
            shape = list(inputs.shape)
            shape[0] = 1
            shape = tuple(shape)
            self.gamma = xp.ones(shape, dtype=float)
            self.beta = xp.zeros(shape, dtype=float)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        self.mean = xp.mean(inputs, axis=-1, keepdims=True)
        self.var = xp.var(inputs, axis=-1, keepdims=True)
        self.x_hat = (inputs - self.mean) / xp.sqrt(self.var + 10e-4)

        return self.gamma * self.x_hat + self.beta

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:

        self.d_beta = dEdO.sum(axis=0, keepdims=True)
        self.d_gamma = xp.sum(self.x_hat * dEdO, axis=0,
                              keepdims=True)

        var_sqrt = xp.sqrt(self.var + 10e-4)

        dX = xp.zeros_like(self._inputs)
        d = dX.shape[1]
        # Nb = dX.shape[0]
        # for n in range(Nb):
        #     s1 = dEdO[n] * self.gamma / var_sqrt[n]
        #     s2 = (dEdO[n] * self.gamma * self.x_hat[n]).sum() * \
        #          self.x_hat[n].sum() * xp.ones((d, )) / (d*d*var_sqrt[n])
        #     s3 = -(dEdO[n] * self.gamma).sum() * xp.ones((d, )) / (d*var_sqrt[n])
        #     s4 = -(dEdO[n] * self.gamma * self.x_hat[n]).sum() * \
        #          self.x_hat[n] / (d*var_sqrt[n])
        #     # s1 = s1.reshape((s1.size, ))
        #     dX[n, :] = s1 + s2 + s3 + s4

        s1 = dEdO * self.gamma / var_sqrt
        s2 = (self.gamma * self.x_hat * dEdO).sum(axis=1, keepdims=True) * self.x_hat.sum(axis=1, keepdims=True) / (d*d * var_sqrt)
        s3 = -(dEdO * self.gamma).sum(axis=1, keepdims=True)/(d*var_sqrt)
        s4 = -(self.gamma * self.x_hat * dEdO).sum(1, keepdims=True) * self.x_hat/(var_sqrt * d)

        return s1 + s2 + s3 + s4

    def update_parameters(self):
        self._optimizer.update_parameters(self.gamma, self.d_gamma)
        self._optimizer.update_parameters(self.beta, self.d_beta)
        self.d_beta = None
        self.d_gamma = None
