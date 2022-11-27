from typing import Tuple

from backend.backend import xp


class Dataset:
    def __init__(self, x: xp.ndarray, y: xp.ndarray,
                 batch_size: int = 32, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.x)

    def __next__(self) -> Tuple[xp.ndarray, xp.ndarray]:
        start_i = self.batch_index * self.batch_size
        end_i = (self.batch_index + 1) * self.batch_size
        if start_i >= len(self.x):
            self.batch_index = 0
            if self.shuffle:
                p = xp.random.permutation(len(self.x))
                self.x, self.y = self.x[p], self.y[p]
            raise StopIteration
        else:
            end_i = min(end_i, len(self.x))
            batch_x = self.x[start_i: end_i]
            batch_y = self.y[start_i: end_i]
            self.batch_index += 1

            return batch_x, batch_y
