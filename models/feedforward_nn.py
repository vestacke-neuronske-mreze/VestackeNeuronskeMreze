import pickle
import time
from typing import List, Tuple, Union

from models.adaptive_object import AdaptiveObject
from backend.backend import xp
from layers.function import Function
from loss_functions.abstract_loss_function import LossFunction
from metrics.metrics import Metric
from optimizers.abstract_optimizer import Optimizer
from utils.dataset import Dataset


class Model(AdaptiveObject):
    def __init__(self, loss_function: LossFunction = None, name: str = "dnn_model"):
        super().__init__(name)
        self._layers: List[Function] = list()
        # jednostavnije smo ovu linjiju mogli zapisati samo kao self._layers = [] ili self._layers = list()
        # deo sa : List[Function] kazuje da ce tip podataka koji ce se smestati u nasu listu layers biti
        # Function (i sve klase izvedene iz Function, naravno). Pycharm ce nam onda nuditi vecu podrsku
        # prilikom kucanja koda.

        self._loss = loss_function
        self._training = False  # da li je mreža završila sa treniranjem ili ne

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        for layer in self._layers:
            layer.training = val

    @property
    def parameters(self) -> list:
        params = list()
        for l in self._layers:
            params.append(l.parameters)
        return params

    @parameters.setter
    def parameters(self, val: tuple):
        for i in range(len(val)):
            self._layers[i].parameters = val[i]

    def save_params(self, filename: str = None):
        if filename is None:
            filename = self.name + ".pickle"

        with open(filename, "wb") as file:
            pickle.dump(self.parameters, file)

        print("Parameters saved at", filename)

    def load_params(self, filename: str):
        with open(filename, "rb") as file:
            self.parameters = pickle.load(file)

    def __call__(self, input_tensor: xp.ndarray) -> xp.ndarray:
        for i in range(len(self._layers)):
            input_tensor = self._layers[i].forward(input_tensor)

        return input_tensor

    def backward(self, dEdY: xp.ndarray) -> xp.ndarray:
        dEdX_next = dEdY
        for i in reversed(range(len(self._layers))):
            dEdX_next = self._layers[i].backward(dEdX_next)

        return dEdX_next

    def update_parameters(self):
        for layer in self._layers:
            if isinstance(layer, AdaptiveObject):
                layer.update_parameters()

    def _process_minibatch(self, x: xp.ndarray,
                           y: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, float]:
        if y.ndim == 1:
            # "formalnost", da bismo radili matrične proizvode,
            # potrebno nam je da shape izlaza bude dvodimenzionalan
            y = y.reshape(-1, 1)

        output = self.forward(x)
        l = self._loss(output, y)
        dEdI = None
        if self.training and self._loss is not None:
            grad = self._loss.backward(output, y)
            dEdI = self.backward(grad)

        return output, dEdI, l

    def _epoch(self, data: Dataset,
               metrics: List[Metric] = []) -> float:
        loss = 0.0
        batch_num = 0
        for batch_x, batch_y in data:
            if batch_y.ndim == 1:
                batch_y = batch_y.reshape(-1, 1)
            batch_num += 1
            output, _, l = self._process_minibatch(batch_x, batch_y)
            if self.training:
                self.update_parameters()
            loss += l  # funkciju greške prikazivaćemo kao prosečnu po svakom primeru

            for m in metrics:
                m.calculate(output, batch_y)

        loss /= batch_num
        for m in metrics:
            m.calculate_for_epoch()
        return loss

    def fit(self,
            train_data: Union[Tuple[xp.ndarray, xp.ndarray], Dataset],
            val_data: Union[Tuple[xp.ndarray, xp.ndarray], Dataset] = None,
            batch_size: int = 64, max_epochs: int = 100,
            print_every: int = 5, eps: float = 1e-6,
            metrics: List[Metric] = []):
        """Treniranje prekidamo nakon najviše max_epochs epoha ili dok razlika između vrednosti funkcije greške u
        dve susedne epohe ne bude manja od eps"""

        if isinstance(train_data, tuple):
            train_data = Dataset(train_data[0], train_data[1])
        train_data.batch_size = batch_size

        if val_data is not None and isinstance(val_data, tuple):
            val_data = Dataset(val_data[0], val_data[1], shuffle=False)

        if val_data is not None:
            val_data.batch_size = batch_size

        prev_loss = None
        prev_val_loss = None

        self.training = True
        start_time = time.time()
        for epoch in range(max_epochs):
            loss = self._epoch(train_data, metrics)
            if (epoch + 1) % print_every == 0:
                self.print_progress(loss, epoch + 1, "Training", metrics)

            if val_data is not None:
                self.training = False
                val_loss = self._epoch(val_data, metrics)
                self.training = True
                if (epoch+1) % print_every == 0:
                    self.print_progress(val_loss, epoch + 1, "Validation", metrics)
                if prev_val_loss is None:
                    prev_val_loss = val_loss
                else:
                    if prev_val_loss < val_loss:
                        self.training = False
                        print("Training time = " + str(time.time() - start_time) + ' seconds')
                        self.save_params()
                        return
                    prev_val_loss = val_loss

            if prev_loss is not None:
                if abs(loss - prev_loss) < eps:
                    self.training = False
                    self.print_progress(loss, epoch + 1, "Training", metrics)
                    print("Training time = " + str(time.time() - start_time) + ' seconds')
                    self.save_params()
                    return

            prev_loss = loss

        self.training = False
        print("Training time = " + str(time.time() - start_time) + ' seconds')
        self.save_params()

    def evaluate(self,
                 test_data: Union[Tuple[xp.ndarray, xp.ndarray], Dataset],
                 metrics: List[Metric] = []) -> float:
        """Funkncija služi za evalucaiju na test skupu."""
        self.training = False

        if isinstance(test_data, tuple):
            test_data = Dataset(test_data[0], test_data[1], 64)
        loss = self._epoch(test_data, metrics)
        print("Test set loss = " + str(loss))

        for metric in metrics:
            m = metric.last_epoch_value()
            print("Metric: {}, value: {}".format(metric.name, m))

        return loss

    def print_progress(self, loss: float, epoch: int, prefix: str = "Training", metrics: List[Metric] = []):
        print("{}:\tEpoch: {}, loss: {}.".format(prefix, epoch, loss))

        for m in metrics:
            print("Metric: {} {} value: {}".format(prefix, m.name, m.last_epoch_value()))
        print("-" * 50 + '\n')

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        for l in self._layers:
            if isinstance(l, AdaptiveObject):
                l.set_optimizer(optimizer, force)

    def add_layer(self, layer: Function):
        self._layers.append(layer)

    def set_loss(self, loss_function: LossFunction):
        self._loss = loss_function
