from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout

from backend.backend import xp
from data_scalers.scalers import MinMaxScaler
from models.feedforward_nn import Model
from layers.activation_functions.relu import ReLU
from layers.activation_functions.sigmoid import Sigmoid
from layers.dense_layer import DenseLayer
from loss_functions.binary_cross_entropy import BinaryCrossEntropy
from optimizers.rmsprop import RMSProp
from utils.dataset import Dataset
from utils.utils import get_mnist_data
from models.vae import VAE


def test_VAE():
    X, y = get_mnist_data(flat_images=True)

    scaler = MinMaxScaler()
    X = scaler.transform(X)
    X = xp.where(X > 0.5, 1.0, 0.0)  # možemo da slike svedemo na bukvalno crno-bele. Samo crni i samo beli pikseli.
    # Kod rekonstrukcije slike odluka je binarna za svaki piksel: da li treba da bude crn ili beo, tj.
    # da li njegova vrednost treba da bude 1 ili 0.
    # u tom slučaju kao kriterijumsku funkciju za dekoder treba izabrati binarnu entropiju.

    z_len = 16
    dense_layer_size = 128
    # Kreiramo dve mreže, enkoder i dekoder
    # Treba voditi računa o dimenzijama izlaza enkodera i ulaza dekodera
    # Izlaz enkodera treba da budu vrednosti gama i mi.
    # Da bismo iskoristili postojeću arhitekturu i klase, bez ikakvog menjanja, enkoder
    # je realizovan tako da vraća izlaz dimenzije 2 x len(z), gde je z latentna slučajna promenljiva.
    # Izlaz tretiramo kao da je prva polovina mi, a druga gama.
    # Više o tome u klasi VAE.
    # Na ovom mestu sada bitno je da imajući to u vidu dimenzija izlaza enkodera i ulaza dekode
    # postavimo da budu u razmeri 2 : 1.
    # Za enkoder mrežu nećemo postavljati kriterijumsku funkciju,
    encoder = Model()
    encoder.add_layer(DenseLayer(X.shape[-1], dense_layer_size))
    encoder.add_layer(ReLU())
    encoder.add_layer(DenseLayer(dense_layer_size, z_len * 2))

    decoder = Model()
    decoder.add_layer(DenseLayer(z_len, dense_layer_size))
    decoder.add_layer(ReLU())
    decoder.add_layer(DenseLayer(dense_layer_size, X.shape[-1]))
    decoder.add_layer(Sigmoid())  # u slučaju da koristimo MSE kao kriterijumsku funkciju
    # za ovaj problem, kod koga su ciljne vrednosti iz [0, 1], možemo kao aktivacionu funkciju da koristimo sigmoidalnu.
    # ali ne mora, može i bez aktivacione.
    # Izlaz dekodera biće 1D vektor dimenzija H*W (za jedan primer, a naravno sa mini batch treniranjem biće Nb x H*W).
    # Taj izlaz treba samo "preoblikovati" da bismo dobili 2D sliku.

    # decoder.set_loss(MSE())
    decoder.set_loss(BinaryCrossEntropy(from_logits=False))
    vae = VAE(encoder, decoder, z_len)
    vae.set_optimizer(RMSProp())
    # 30 epoha je sasvim dovoljno. Štaviše, već sa 10 su rezultati solidni.
    num_of_epochs = 30
    vae.fit(Dataset(X, X), print_every=1, batch_size=100, max_epochs=num_of_epochs)
    # vae.load_params("VAE.pickle")
    # Po završetku treniranja uzećemo num_of_test_samples slika iz skupa uzoraka za treniranje i propustiti ih kroz
    # varijacioni autoenkoder. Originalne primere i dobijene rezultate prikazaćemo na grafiku.
    num_of_test_samples = 10
    generate_random_samples = False
    # for k in range(5):
    # vae.decoder.add_layer(Sigmoid())
    xp.random.seed(11)
    xp.random.shuffle(X)
    if generate_random_samples:
        s = vae.generate_new_samples(num_of_samples=num_of_test_samples)
    else:
        s = vae.generate_new_samples(num_of_samples=num_of_test_samples, samples_like=X[:num_of_test_samples])
    im_dim = int(xp.sqrt(X.shape[1]))  # jer su slike kvadratne, ali ispravljene u niz
    s = s.reshape((-1, im_dim, im_dim))

    # Ostatak koda tiče se prikaza slika na grafiku.
    nrows = 1 if generate_random_samples else 2
    fig, axs = plt.subplots(nrows=nrows, ncols=num_of_test_samples, figsize=(16, 4),
                            subplot_kw={'xticks': [], 'yticks': []})
    # fig.suptitle('VAE test_examples, num of classes = {}, num of epochs = {}'.format(num_of_classes, num_of_epochs))

    if "cupy" in str(xp._version):
        X = xp.asnumpy(X)
        s = xp.asnumpy(s)

    images = []
    for j in range(num_of_test_samples):
        if generate_random_samples:
            images.append(axs[j].imshow(s[j], interpolation='nearest'))
            axs[j].label_outer()

        else:
            images.append(axs[0, j].imshow(X[j].reshape(im_dim, im_dim), interpolation='nearest'))
            images.append(axs[1, j].imshow(s[j], interpolation='nearest'))
            axs[0, j].label_outer()
            axs[1, j].label_outer()

    tight_layout()
    plt.show()
