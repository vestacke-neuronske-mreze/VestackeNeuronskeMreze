import pandas as pd

from data_scalers.scalers import *
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.dense_layer import DenseLayer
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from metrics.metrics import BinaryAccuracy
from optimizers.sgd import SGD
from utils.dataset import Dataset


def test_binary_classification():
    dataset_name = "banknote_authentication"
    dataset_name = "heart"

    file = pd.read_csv('data/' + dataset_name + '.csv')  # učitaćemo vrednosti iz csv fajla pomoću pandas biblioteke
    # Fajl sadrži podatke o pacijentima i sadrži kolonu target koja predstavlja labelu kategorije uzorka.
    # Svaki uzorak može se naći u jednoj od dve kategorije: 1 - ima većih šansi za srčani udar ili 0 - male su šanse za srčani udar
    # Detalje o skupu možete naći na: https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility

    # file = pd.read_csv('data/data_banknote_authentication.csv')  # drugi primer: klasifikacija novčanica u originalne ili falsifikate
    # preuzeto sa http://archive.ics.uci.edu/ml/datasets/banknote+authentication

    data = xp.array(file.values)
    xp.random.shuffle(data)
    N = len(data)

    training_data_ratio = 0.7
    training_data_size = int(N * training_data_ratio)

    y = data[:, -1]
    X = data[:, :-1]

    train_X, test_X = X[: training_data_size], X[training_data_size:]
    train_y, test_y = y[: training_data_size], y[training_data_size:]

    scaler_x = MinMaxScaler()

    scaler_x.adapt(train_X)

    train_X = scaler_x.transform(train_X)
    test_X = scaler_x.transform(test_X)

    m = len(test_X)//2
    val_X, test_X = test_X[:m], test_X[m:]
    val_y, test_y = test_y[:m], test_y[m:]

    test_data = Dataset(test_X, test_y, shuffle=False)
    val_data = Dataset(val_X, val_y, shuffle=False)
    train_data = Dataset(train_X, train_y)

    model = Model(name="bin_class_" + dataset_name)
    model.add_layer(DenseLayer(train_X.shape[1], 64,
                               name='Dense layer 1'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(64, 32, name='Dense layer 2'))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(32, 1, name='Dense layer 3'))
    from_logits = True

    # model.add_layer(Sigmoid())
    # from_logits = False

    loss = BinaryCrossEntropy(from_logits=from_logits)
    model.set_loss(loss)
    model.set_optimizer(SGD(lr=0.1))

    model.fit(train_data, val_data, print_every=1, batch_size=32,
              max_epochs=50, metrics=[BinaryAccuracy(from_logits)])

    model.evaluate(test_data,
                   metrics=[BinaryAccuracy(from_logits)])
