from enum import Enum

from backend.backend import xp
from sklearn.datasets import load_iris, load_wine

from data_scalers.scalers import MinMaxScaler
from models.feedforward_nn import Model
from layers.activation_functions.leakyReLU import LeakyReLU
from layers.dense_layer import DenseLayer
from layers.normalizations.layer_normalization import LayerNormalization
from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from optimizers.adam import Adam
from utils.dataset import Dataset
from utils.utils import get_mnist_data


class UseDataset(Enum):
    MNIST = 1
    WINE = 2
    IRIS = 3


def test_multiclass_classification(use_dataset: UseDataset = UseDataset.MNIST):
    # use_dataset = UseDataset.MNIST
    if use_dataset == UseDataset.MNIST:
        X, y_one_hot = get_mnist_data(flat_images=True)
        num_of_classes = y_one_hot.shape[-1]
        max_epochs = 15
    else:
        max_epochs = 150
        if use_dataset == UseDataset.IRIS:
            data = load_iris()  # više o skupu uzoraka za treniranje na: https://archive.ics.uci.edu/ml/datasets/iris
        else:
            data = load_wine()  # više o skupu uzoraka za treniranje na: https://archive.ics.uci.edu/ml/datasets/Wine

        X = xp.array(data['data'])
        y = xp.array(data['target'])
        # batch_size = 64
        # max_epochs = 300
        print(y)  # vidimo da klase nisu izmešane

        # potrebno je izmešati redosled elemenata u oba niza, ali ipak i-ti red iz matrice X mora da odgovara i-tom elementu
        # vektora y

        # # jedan od načina da to postignemo je sledeći:
        xp.random.seed(123)  # postavimo bilo koji seed vrednost
        xp.random.shuffle(X)
        xp.random.seed(123)  # vratimo ponovo istu seed vrednost!
        xp.random.shuffle(y)

        num_of_classes = int(xp.max(y) + 1)  # klase su označene brojevima 0, 1, ... , num_of_classes - 1
        # y_one_hot = y.reshape((-1, 1))
        y_one_hot = xp.zeros((len(y), num_of_classes), dtype=float)
        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1

    N = len(X)
    training_data_ratio = 0.8
    # Sada treba da podatke podelimo na training i test skup
    m = int(N * training_data_ratio)

    train_X, test_X = X[: m], X[m: ]
    train_y, test_y = y_one_hot[: m], y_one_hot[m: ]

    scaler_x = MinMaxScaler()
    scaler_x.adapt(train_X)

    train_X = scaler_x.transform(train_X)
    test_X = scaler_x.transform(test_X)

    test_data = Dataset(test_X, test_y, batch_size=16)
    train_data = Dataset(train_X, train_y)

    normalization = LayerNormalization
    # normalization = BatchNormalization
    model = Model(name="multiclass_" + str(use_dataset.name))
    model.add_layer(DenseLayer(train_X.shape[1], 64, name='Dense layer 1'))
    # model.add_layer(normalization())
    model.add_layer(LeakyReLU())
    model.add_layer(DenseLayer(64, 32, name='Dense layer 2'))
    # model.add_layer(normalization())
    model.add_layer(LeakyReLU())
    model.add_layer(DenseLayer(32, num_of_classes, name='Dense layer 3'))

    # naredne dve linije idu zajedno. Ili ćemo koristiti eksplicitno softmax sloj i postaviti parametar with_softmax na False
    # ili ćemo prosto samo dodati kros entropiju kao kriterijumsku funkciju, a bez dodavanja aktivacionog sloja sa softmax aktiv.
    # funkcijom (preporučeni pristup)

    # network.add_layer(Softmax())
    # network.set_loss_function(CrossEntropy(with_softmax=False))

    # ili, možemo da prethodne dve linije stavimo u komentar i iskoristimo narednu liniju:
    model.set_loss(CrossEntropy(from_logits=True, one_hot=True))

    model.set_optimizer(Adam(nesterov=True))
    model.fit(train_data, print_every=2, batch_size=64,
              max_epochs=max_epochs, metrics=[Accuracy(one_hot=True)])
    model.evaluate(test_data, metrics=[Accuracy(one_hot=True)])
