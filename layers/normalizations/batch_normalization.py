from models.adaptive_object import AdaptiveObject
from backend.backend import xp


class BatchNormalization(AdaptiveObject):

    def __init__(self, alpha: float = 0.99, name: str = 'Batch normalization layer'):
        super().__init__(name)

        self.batch_mean = None
        self.batch_var = None
        self.mean = None
        self.var = None
        self.gamma = None
        self.d_gamma = None
        self.beta = None
        self.d_beta = None
        self.x_hat = None
        self.alpha = alpha

    @property
    def parameters(self) -> tuple:
        return self.gamma, self.beta, self.mean, self.var

    @parameters.setter
    def parameters(self, val: tuple):
        self.gamma, self.beta, self.mean, self.var = val

    def _pre_fw(self, inputs: xp.ndarray):
        super()._pre_fw(inputs)
        if self.mean is None:
            shape = list(inputs.shape)
            shape[0] = 1
            self.mean = xp.zeros(shape, dtype=float)
            self.var = xp.zeros(shape, dtype=float)
            self.gamma = xp.ones(shape, dtype=float)
            self.beta = xp.zeros(shape, dtype=float)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        if not self._training:
            return self.gamma * (inputs - self.mean) \
                   / xp.sqrt(self.var + 10e-4) + self.beta
        else:
            self.batch_mean = xp.mean(inputs, axis=0, keepdims=True)
            self.batch_var = xp.var(inputs, axis=0, keepdims=True)

            self.mean = self.alpha * self.mean + (1 - self.alpha) * self.batch_mean
            self.var = self.alpha * self.var + (1 - self.alpha) * self.batch_var

            self.x_hat = (inputs - self.batch_mean) / xp.sqrt(self.batch_var + 10e-4)

            return self.gamma * self.x_hat + self.beta

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        Nb = self._inputs.shape[0]

        self.d_beta = dEdO.sum(axis=0, keepdims=True)
        self.d_gamma = xp.sum(self.x_hat * dEdO, axis=0, keepdims=True)

        _tmp_array = xp.add(self.batch_var, 1e-4)
        xp.sqrt(_tmp_array, out=_tmp_array)
        var_sqrt = xp.sqrt(self.batch_var + 1e-4)

        s1 = self.gamma * dEdO / var_sqrt
        s2 = (dEdO * self.x_hat).sum(axis=0, keepdims=True) * \
             self.x_hat.sum(axis=0, keepdims=True) * \
             self.gamma / (Nb * Nb * var_sqrt)
        s3 = - 1 / Nb * self.gamma / var_sqrt * dEdO.sum(axis=0, keepdims=True)
        s4 = - 1 / Nb * self.x_hat * self.gamma / var_sqrt * \
             xp.sum(self.x_hat * dEdO, axis=0, keepdims=True)

        return s1 + s2 + s3 + s4

    def update_parameters(self):
        self._optimizer.update_parameters(self.gamma, self.d_gamma)
        self._optimizer.update_parameters(self.beta, self.d_beta)
        self.d_beta = None
        self.d_gamma = None

