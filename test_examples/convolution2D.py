from data_scalers.scalers import MinMaxScaler
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.conv_layer.conv_layer_algorithms import *
from layers.conv_layer.convolution_layer import Convolution2D
from layers.conv_layer.pooling import Pooling
from layers.dense_layer import DenseLayer
from layers.flattening_layer import FlatteningLayer
from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from optimizers.rmsprop import RMSProp
from utils.dataset import Dataset
from utils.utils import get_mnist_data


def test_CNN(use_8x8_images: bool = False):
    model = Model()
    if use_8x8_images:
        max_epochs = 10
    else:
        max_epochs = 10

    X, y = get_mnist_data(_8x8=use_8x8_images)

    training_data_ratio = 0.8
    scaler = MinMaxScaler()
    X = scaler.transform(X)

    N = len(X)
    num_of_classes = y.shape[-1]
    training_data_size = int(N * training_data_ratio)

    train_X, test_X = X[: training_data_size], X[training_data_size:]
    train_y, test_y = y[: training_data_size], y[training_data_size:]

    algorithm = Matmul
    model.add_layer(Convolution2D(X.shape[-3], 6, 5, algorithm=algorithm()))
    model.add_layer(Pooling(6, type="average"))
    model.add_layer(ReLU())

    model.add_layer(Convolution2D(6, 16, 7, 2, 2, algorithm=algorithm()))
    model.add_layer(ReLU())
    tmp = model(X[0:1])

    model.add_layer(FlatteningLayer())
    model.add_layer(DenseLayer(tmp.size, 16, name='Dense layer 1'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(16, 32, name='Dense layer 2'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(32, num_of_classes, name='Dense layer 3'))

    model.set_loss(CrossEntropy())

    model.set_optimizer(RMSProp())

    model.fit(Dataset(train_X, train_y), print_every=1, batch_size=50, max_epochs=max_epochs, metrics=[Accuracy()])
    model.evaluate(Dataset(test_X, test_y), metrics=[Accuracy()])
