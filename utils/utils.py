import string
from typing import Union, Optional, List

import pandas as pd
from sklearn.datasets import load_digits

from backend.backend import xp


def generate_vocabulary(text: str,
                        padding: str = None) -> (dict, dict):
    chars = sorted(list(set(text)))
    if padding is not None and padding not in chars:
        chars.append(padding)

    return dict((c, i) for i, c in enumerate(chars)), \
           dict((i, c) for i, c in enumerate(chars))


def get_one_hot_vector(size: int, i: int) -> List[int]:
    x = [0] * size
    x[i] = 1
    return x


def generate_sequences(text: str, seq_delimiter: Optional[str] = '\n',
                       seq_length: Union[int, str] = 'auto', padding: str = ".") -> (xp.ndarray, xp.ndarray):
    char_to_int, int_to_char = generate_vocabulary(text, padding)

    data_X = []
    data_y = []
    vocab_size = len(char_to_int)

    if seq_delimiter is None and seq_length == "auto":
        raise Exception("Either seq_length or sequence_delimiter must be specified!")

    if seq_delimiter is None:
        num_of_sequences = len(text) // seq_length
        for seq_i in range(num_of_sequences):
            new_seq_x = []
            new_seq_y = []
            for i in range(seq_length):
                global_idx = seq_i * seq_length + i
                next_char = text[global_idx + 1]
                curr_char = text[global_idx]
                new_seq_y.append(get_one_hot_vector(vocab_size, char_to_int[next_char]))
                new_seq_x.append(get_one_hot_vector(vocab_size, char_to_int[curr_char]))
            data_X.append(new_seq_x)
            data_y.append(new_seq_y)
    else:
        sequences = text.split(sep=seq_delimiter)
        if seq_length == 'auto':
            seq_length = len(sequences[0])
            for i in range(1, len(sequences)):
                seq_length = max(seq_length, len(sequences[i]))
        for seq in sequences:
            if len(seq) < seq_length + 1:
                seq += padding * (1 + seq_length - len(seq))  # right padding
                # seq = '.' * (seq_length - len(seq)) + seq + "."  # left padding

                # center padding
                # seq = '.' * ((1 + seq_length - len(seq)) // 2) + seq + '.' * ((1 + seq_length - len(seq)) // 2)
                # if len(seq) < seq_length + 1:
                #     seq += '.'
                # seq = seq[:seq_length+1]

            new_seq_x = []
            new_seq_y = []

            for char_index in range(seq_length):
                next_char = seq[char_index + 1]
                curr_char = seq[char_index]
                new_seq_y.append(get_one_hot_vector(vocab_size, char_to_int[next_char]))
                new_seq_x.append(get_one_hot_vector(vocab_size, char_to_int[curr_char]))

            data_X.append(new_seq_x)
            data_y.append(new_seq_y)

    return xp.array(data_X), xp.array(data_y)


def read_txt_books() -> str:
    import glob
    import os
    path = '../data/books/'
    books_content = ''
    for filename in glob.glob(os.path.join(path, '*.txt')):
        raw_text = open(filename, 'r', encoding='utf-8').read()
        start_idx = raw_text.find('START OF')
        while True:
            c = raw_text[start_idx]
            start_idx += 1
            if c == '\n':
                break
        raw_text = raw_text[start_idx:]

        end_idx = raw_text.find('END OF THE')
        while True:
            c = raw_text[end_idx]
            end_idx -= 1
            if c == '\n':
                break
        raw_text = raw_text[:end_idx]
        raw_text = raw_text.lower()

        books_content += raw_text

    return books_content


def get_news_data(news_website: str, news_count: int = 100):
    import newspaper
    from newspaper import Article

    site = newspaper.build(news_website, memoize_articles=False)

    news_urls = site.article_urls()
    N = min(news_count, len(news_urls))
    content = ''
    for i in range(N):
        try:
            article = Article(news_urls[i])
            article.download()
            article.parse()
            txt = article.text
            content += txt.translate(str.maketrans('', '', string.punctuation))
            # content += str(txt)
            content += '\n_\n'
        except:
            print('error')

    return content


def get_cifar_10_data(file: str) -> (xp.ndarray, xp.ndarray):
    import pickle
    with open(file, 'rb') as fo:
        d: dict = pickle.load(fo, encoding='bytes')

    X = xp.array(d[b'data'], dtype=int)
    X = xp.reshape(X, (X.shape[0], 3, 32, 32))

    labels: list = d[b'labels']
    num_of_classes = len(set(labels))
    y = xp.zeros((len(labels), num_of_classes))
    for label_index in range(len(labels)):
        label = labels[label_index]
        y[label_index, label] = 1

    return X, y


def get_8x8_mnist_data(flat_images: bool = False, max_num_of_samples: int = None, max_num_of_classes: int = None) -> (xp.ndarray, xp.ndarray):
    data = load_digits()  # više o skupu uzoraka za treniranja na:
    # https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html#sphx-glr-auto-examples-datasets-plot-digits-last-image-py

    X_flat = data['data']
    t = data['target']

    num_of_classes = 10

    data_numpy = xp.hstack((t.reshape(-1, 1), X_flat))

    if max_num_of_classes is not None:
        num_of_classes = max_num_of_classes
        data_numpy = data_numpy[data_numpy[:, 0] < num_of_classes]

    y = data_numpy[:, 0]
    X = data_numpy[:, 1:]
    X = X.reshape((len(X), 1, 8, 8))

    y = xp.zeros((len(X), num_of_classes))

    for label_index in range(len(data_numpy)):
        label = int(data_numpy[label_index, 0])
        y[label_index, label] = 1

    if flat_images:
        X = X.reshape(X.shape[0], -1)

    if max_num_of_samples is not None:
        if max_num_of_samples < len(X):
            X = X[:max_num_of_samples]
            y = y[:max_num_of_samples]

    xp.random.seed(123)
    xp.random.shuffle(X)
    xp.random.seed(123)
    xp.random.shuffle(y)

    return X, y


def to_one_hot(t: xp.ndarray, num_of_classes: int):
    tmp = xp.zeros((t.size, num_of_classes), dtype=float)
    tmp[xp.arange(0, t.size), t.astype(int).squeeze()] = xp.ones(t.size)
    return tmp


def get_mnist_data(flat_images: bool = False, max_num_of_samples: int = None, max_num_of_classes: int = None, _8x8: bool = False) -> (xp.ndarray, xp.ndarray):

    if _8x8:
        data = load_digits()  # više o skupu uzoraka za treniranja na:
        # https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html#sphx-glr-auto-examples-datasets-plot-digits-last-image-py
        X_flat = data['data']
        t = data['target']
        data_numpy = xp.hstack((t.reshape(-1, 1), X_flat))
        im_size = 8
    else:
        mnist_dataframe = pd.read_csv('data/mnist_csvs/mnist_train.csv')
        data_numpy = xp.array(mnist_dataframe.to_numpy())
        im_size = 28

    num_of_classes = 10
    if max_num_of_classes is not None:
        num_of_classes = max_num_of_classes
        data_numpy = data_numpy[data_numpy[:, 0] < num_of_classes]

    # y = data_numpy[:, 0]
    X = data_numpy[:, 1:]
    X = X.reshape((len(X), 1, im_size, im_size))

    y = xp.zeros((len(X), num_of_classes))

    for label_index in range(len(data_numpy)):
        label = int(data_numpy[label_index, 0])
        y[label_index, label] = 1

    if flat_images:
        X = X.reshape(X.shape[0], -1)

    if max_num_of_samples is not None:
        if max_num_of_samples < len(X):
            X = X[:max_num_of_samples]
            y = y[:max_num_of_samples]

    xp.random.seed(123)
    xp.random.shuffle(X)
    xp.random.seed(123)
    xp.random.shuffle(y)

    return X, y
