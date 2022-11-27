from abc import ABC, abstractmethod

from backend.backend import xp


class Scaler(ABC):
    @abstractmethod
    def adapt(self, data: xp.ndarray):
        pass

    @abstractmethod
    def transform(self, data: xp.ndarray) -> xp.ndarray:
        pass

    @abstractmethod
    def inverse(self, data: xp.ndarray) -> xp.ndarray:
        pass


class MinMaxScaler(Scaler):
    def __init__(self, new_min: float = 0.0, new_max: float = 1.0):
        self.new_min = new_min
        self.new_max = new_max
        self.min = None
        self.max = None

    def adapt(self, data: xp.ndarray):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data: xp.ndarray) -> xp.ndarray:
        if self.max is None:
            self.adapt(data)

        return (self.new_max - self.new_min) * (data - self.min) / \
               (xp.maximum(self.max - self.min, 1e-4)) \
               + self.new_min

    def inverse(self, data: xp.ndarray) -> xp.ndarray:
        if self.min is None:
            raise Exception("Adapt not called!")

        return (self.max - self.min) * (data - self.new_min) / \
               (self.new_max - self.new_min) + self.min


class StandardScaler(Scaler):
    def __init__(self):
        self.mean = None
        self.std = None

    def adapt(self, data: xp.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data: xp.ndarray) -> xp.ndarray:
        if self.mean is None:
            self.adapt(data)

        return (data - self.mean) / (self.std + 1e-8)

    def inverse(self, data: xp.ndarray) -> xp.ndarray:
        if self.mean is None:
            raise Exception("Adapt not called!")

        return data * self.std + self.mean


class SigmoidScaler(Scaler):

    def transform(self, data: xp.ndarray) -> xp.ndarray:
        return 1 / (1 + xp.exp(-1*data))

    def inverse(self, transformed_data: xp.ndarray) -> xp.ndarray:
        return xp.log(transformed_data) - xp.log(1 - transformed_data)


if __name__ == '__main__':
    test_matrix = xp.array([1, 3, 4, -10, 3, -4, 0, 6, 2, -1]).reshape((5, 2))
    xp.random.shuffle(test_matrix)
    # scaler = MinMaxScaler()
    # scaler = StandardizationScaler()
    scaler = SigmoidScaler()
    transformed_data = scaler.transform(test_matrix)
    print(test_matrix)
    print(transformed_data)
    test_matrix = scaler.inverse(transformed_data)
    print(test_matrix)