from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout

from backend.backend import xp
from data_scalers.scalers import MinMaxScaler
from models.feedforward_nn import Model
from models.gan import GAN
from layers.activation_functions.leakyReLU import LeakyReLU
from layers.activation_functions.relu import ReLU
from layers.activation_functions.sigmoid import Sigmoid
from layers.dense_layer import DenseLayer
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from metrics.metrics import BinaryAccuracy
from optimizers.rmsprop import RMSProp
from utils.utils import get_mnist_data


def test_GAN():
    X, y = get_mnist_data(flat_images=True)

    scaler = MinMaxScaler()
    X = scaler.transform(X)

    z_len = 16
    discr_dense_l = 64
    gen_dense_l = 128
    # Kreiramo dve mreže, generator i diskriminator
    # Slično kao kod varijacionog autoenkodera, treba voditi računa o dimenzijama izlaza i ulaza dve mreže.
    # Ulaz u generator je slučajni šum, a izlaz treba da po dimenzijama bude identičan primerima iz skupa uzoraka
    # za treniranje.
    # Ulaz u diskriminator mora da bude dimenzija kao izlaz generatora (što se naravno poklapa sa dimenzijama
    # podataka iz skupa uzoraka za treniranje).
    # Izlaz diskriminatora je verovatnoća da uzorak dolazi iz prave, originalne, raspodele podataka.
    discr = Model(name="discriminator")
    discr.add_layer(DenseLayer(X.shape[-1], discr_dense_l))
    discr.add_layer(LeakyReLU())
    discr.add_layer(DenseLayer(discr_dense_l, 1))
    discr.add_layer(Sigmoid())
    discr.set_loss(BinaryCrossEntropy(from_logits=False))

    gen = Model(name="generator")
    gen.add_layer(DenseLayer(z_len, gen_dense_l))
    gen.add_layer(ReLU())
    gen.add_layer(DenseLayer(gen_dense_l, X.shape[-1]))
    gen.add_layer(ReLU())

    gan = GAN(gen, discr, z_len, 1)
    gan.set_optimizer(RMSProp())
    gan.fit((X, X), print_every=1, batch_size=100, max_epochs=30, metrics=[BinaryAccuracy(from_logits=False)])

    num_of_test_samples = 10
    s = gan.generate_new_samples(num_of_samples=num_of_test_samples)

    im_dim = int(xp.sqrt(X.shape[1]))  # jer su slike kvadratne, ali ispravljene u niz
    s = s.reshape((-1, im_dim, im_dim))

    if "cupy" in str(xp._version):
        s = xp.asnumpy(s)

    # Ostatak koda tiče se prikaza slika na grafiku.
    nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=num_of_test_samples, figsize=(16, 4),
                            subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle('GAN test_examples')

    images = []
    for j in range(num_of_test_samples):
        images.append(axs[j].imshow(s[j], interpolation='nearest'))
        axs[j].label_outer()

    tight_layout()
    plt.show()
