from models.feedforward_nn import Model
from layers.dense_layer import DenseLayer
from layers.recurrent_layers.gru import GRU
from layers.recurrent_layers.lstm import LSTM
from loss_functions.cross_entropy import CrossEntropy
from metrics.metrics import Accuracy
from optimizers.adam import Adam
from utils.dataset import Dataset
from utils.utils import generate_vocabulary, generate_sequences
from backend.backend import xp


def test_LSTM_GRU():
    model = Model(name="LSTM_GRU")

    # news_url = 'https://www.danas.rs'
    # # news_url = "https://www.bbc.com/"
    # content_str = get_news_data(news_url, 350)
    # seq_delimiter = None
    # seq_len = 75

    content_str = open('data/dino.txt').read()
    seq_len = "auto"
    seq_delimiter = '\n'

    content_str = content_str.lower()
    # print(content_str)
    # linijom ispod dobijamo dva python dictionary objekta
    # Jedan mapira svako od slova u int, a drugi radi inverzno mapiranje.
    char_to_int, int_to_char = generate_vocabulary(content_str, '/')
    # generate_sequences nam vraća xp.ndarray nizove X i y koje ćemo koristiti
    # za treniranje.
    # Karakteri u oba niza su predstavljeni pomoću one-hot enkodiranja,
    # A dimenzije oba niza su Ns x T x D, gde je Ns broj sekvenci dužine T pri čemu je D broj različirih
    # mogućih karaktera (Ns x T <= N, gde je N ukupan broj karaktera u celokupnom tekstu)
    X, y = generate_sequences(content_str, seq_length=seq_len, seq_delimiter=seq_delimiter, padding='/')

    model.add_layer(DenseLayer(X.shape[-1], 32))
    model.add_layer(LSTM(32, 256))
    # model.add_layer(GRU(256, 128))

    model.add_layer(DenseLayer(256, len(char_to_int), name='Dense layer 1'))

    model.set_loss(CrossEntropy())

    model.set_optimizer(Adam())
    batch_size = 50
    model.fit(Dataset(X, y), print_every=1, batch_size=batch_size, max_epochs=50, metrics=[Accuracy()])
    # model.load_params("LSTM_GRU.pickle")
    xp.random.shuffle(X)

    for l in model._layers:
        if isinstance(l, LSTM) or isinstance(l, GRU):
            l.reset_state = False

    network_input = xp.zeros((batch_size, 1, len(char_to_int)))
    # generisaćemo batch_size stringova, a linijom ispod dobijamo listu dužine batch_size koja
    # je ispunjena praznim stringovima
    generated_text = [''] * batch_size
    print()

    warm_up = 3  # koliko ćemo slova dati mreži da izračuna stanje rekurentnih slojeva
    # U konkretnom primeru, prvih 4 karaktera svakog generisanog teksta biće iz training skupa
    # a nastavak stringa je generisan od strane mreže
    for t in range(warm_up):
        for i in range(batch_size):
            network_input[i, 0, :] = X[i, t, :]
            c = int_to_char[int(xp.argmax(network_input[i, 0, :]))]
            generated_text[i] += c
        network_output = model.forward(network_input)

    text_len = 50
    for t in range(text_len):
        for i in range(batch_size):
            # xp.random.choice() - bira nasumični element niza sa verovatnoćama odabira svakog
            # od njih datom pomoću parametra p. Mi prosleđujemo kao niz range(len(char_to_int)), tj.
            # niz 0, 1, 2, ... , len(char_to_int) - 1
            # c_idx = int(xp.random.choice(range(len(char_to_int)), p=Softmax()(network_output[i, 0, :]), size=1))
            c_idx = int(xp.argmax(network_output[i, 0, :]))
            generated_text[i] += int_to_char[c_idx]  # pretvaramo dobijeni broj u karakter i konkatenišemo na generisani tekst
            network_input[i] = 0  # network_input ćemo i dalje koristiti kao ulaz u mrežu. Ulaz je one-hot enkodiran, pa
            # treba jedinicu postaviti na odgovarajuće mesto.
            # Najlakše je da prvo sve postavimo na nule, a zatim podesimo jedinicu
            network_input[i, 0, c_idx] = 1
        network_output = model.forward(network_input)

    for i in range(batch_size):
        print(generated_text[i])
        print('-----------------------')