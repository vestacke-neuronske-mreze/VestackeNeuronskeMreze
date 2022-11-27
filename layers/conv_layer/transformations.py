from typing import Tuple

from backend.backend import xp


def im2col_map(Xcol_shape: tuple, f_h: int, f_w: int,
               stride: int, out_w: int) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
    n, i, j = xp.indices(Xcol_shape)

    dl = i // (f_w * f_h)
    f_i = (i % (f_w * f_h)) // f_w
    f_j = (i % (f_w * f_h)) % f_w

    # A šta nam sve 'j' otkriva?
    # 'j' je indeks po dimenziji veličine Oh * Ow i hoćemo da se u j-toj koloni nađu svi oni ulazi koje treba množiti da bismo
    # dobili j-ti piksel izlazne mape karakteristika (j-ti ako bismo gledali mapu kao niz).
    # Dakle, iz j možemo da dobijemo 2D indekse u izlaznoj mapi karakteristika za čije računanje je j-ta kolona zadužena.
    # To lako radimo znajući dimenzije izlaznih mapa karakteristika, tj. širina je dovoljna.

    i_out = j // out_w
    j_out = j % out_w

    # Sada znamo sve, skoro. Znamo da će element na mestu output_flat_index biti takav element iz X da pripada primeru batch_idx,
    # da je iz ulazne mape karakteristika sa indeksom dl (znamo to jer ga množi filter koji je zadužen za tu mapu),
    # i da pomoću polja filtera f_i, f_j utiče da se izračuna aktivacioni potencijal na mestu i_out, j_out.
    # Sada samo treba proračunati koji je to tačno element u ulaznoj mapi karakteristika...

    i_in = i_out * stride + f_i  # Imati na umu da su i_in i in_j indeksi ulazne slike kojoj je dodat padding!
    j_in = j_out * stride + f_j

    return n, dl, i_in, j_in


def col2im(dEdXcol: xp.ndarray, X_shape: tuple, kernel_h: int, kernel_w: int, stride: int):
    dEdX = xp.zeros(X_shape, dtype=float)
    out_h = (X_shape[-2] - kernel_h) // stride + 1
    dEdX[im2col_map(dEdXcol.shape, kernel_h, kernel_w, stride, out_h)] += dEdXcol

    return dEdX


def im2col(X: xp.ndarray, kernel_h: int, kernel_w: int, stride: int) -> xp.ndarray:
    out_w = (X.shape[-1] - kernel_w) // stride + 1
    out_h = (X.shape[-2] - kernel_h) // stride + 1

    xcol_shape = (X.shape[0], X.shape[1] * kernel_h * kernel_w, out_h * out_w)
    return X[im2col_map(xcol_shape, kernel_h, kernel_w, stride, out_w)].reshape(xcol_shape)
