from layers.conv_layer.conv_layer_algorithms import *
from layers.function import Function


class Pooling(Function):

    @staticmethod
    def get_height(input_height: int, kernel_height: int, padding: int, stride: int) -> int:
        """Znajući visinu ulaza, visinu kernela (fitlera), padding i stride, možemo izračunati visinu izlaza"""
        return (input_height - kernel_height + 2 * padding) // stride + 1

    @staticmethod
    def get_width(input_width: int, kernel_width: int, padding: int, stride: int) -> int:
        """Znajući širinu ulaza, širinu kernela (fitlera), padding i stride, možemo izračunati širinu izlaza"""
        return (input_width - kernel_width + 2 * padding) // stride + 1

    def __init__(self, in_maps_n: int, kernel_size: int = 3,
                 padding: int = 0, stride: int = 1, type="max",
                 name: str = 'Pooling Layer'):
        super().__init__(name)
        self.in_maps_n = in_maps_n
        self.pad = padding
        self.s = stride
        self.size = kernel_size
        self.type = type
        if type == "max":
            self.xp_f = xp.max
        else:
            self.xp_f = xp.mean

    def _add_padding(self, X: xp.ndarray) -> xp.ndarray:
        if self.pad == 0:
            return X
        return xp.pad(X, ((0, 0), (0, 0), (self.pad, self.pad),
                      (self.pad, self.pad)), mode='constant')

    def _remove_padding(self, X: xp.ndarray) -> xp.ndarray:
        if self.pad == 0:
            return X
        return X[:, :, self.pad: - self.pad, self.pad: - self.pad]

    def __call__(self, X: xp.ndarray) -> xp.ndarray:
        Nb = X.shape[0]
        D = self.in_maps_n
        H = Pooling.get_height(X.shape[-2], self.size, self.pad, self.s)
        W = Pooling.get_width(X.shape[-1], self.size, self.pad, self.s)
        Y = xp.zeros((Nb, D, H, W))

        X = self._add_padding(X)

        for i in range(H):
            for j in range(W):
                i_s = i*self.s
                j_s = j*self.s

                i_e = i_s + self.size
                j_e = j_s + self.size
                X_slice = X[:, :, i_s:i_e, j_s:j_e]
                Y[:, :, i, j] = self.xp_f(X_slice, axis=(2, 3))

        return Y

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:

        X = self._add_padding(self._inputs)
        dEdX = xp.zeros_like(X)
        Nb = X.shape[0]
        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        for i in range(output_h):
            for j in range(output_w):
                i_s = i * self.s
                j_s = j * self.s

                i_e = i_s + self.size
                j_e = j_s + self.size

                X_slice = X[:, :, i_s:i_e, j_s:j_e]
                if self.type == "max":
                    max = xp.max(X_slice, axis=(2, 3)).reshape((Nb,-1,1,1))
                    bit_mask = xp.equal(X_slice, max)
                else:
                    bit_mask = 1
                dEdO_slice = dEdO[:, :, i, j].reshape((Nb, -1, 1, 1))
                dEdX[:, :, i_s: i_e, j_s: j_e] += bit_mask * dEdO_slice

        dEdX = self._remove_padding(dEdX)

        return dEdX
