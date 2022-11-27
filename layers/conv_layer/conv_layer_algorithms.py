from abc import ABC, abstractmethod
from typing import Tuple

from backend.backend import xp
from layers.conv_layer.transformations import im2col, col2im


class Conv2DAlgo(ABC):

    def __init__(self, padding: int = 0, stride: int = 1):
        self.pad = padding
        self.s = stride

    def _output_shape(self, X_shape: tuple, W_shape: tuple) -> tuple:
        from layers.conv_layer.convolution_layer import Convolution2D
        h = Convolution2D.get_height(X_shape[-2], W_shape[-2], self.pad, self.s)
        w = Convolution2D.get_width(X_shape[-1], W_shape[-1], self.pad, self.s)
        output_shape = (X_shape[0], W_shape[0], h, w)
        return output_shape

    def _add_padding(self, X: xp.ndarray) -> xp.ndarray:
        if self.pad == 0:
            return X
        return xp.pad(X, ((0, 0), (0, 0), (self.pad, self.pad),
                      (self.pad, self.pad)), mode='constant')

    def _remove_padding(self, X: xp.ndarray) -> xp.ndarray:
        if self.pad == 0:
            return X
        return X[:, :, self.pad: - self.pad, self.pad: - self.pad]

    def prepare(self, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        Y = xp.zeros(self._output_shape(X.shape, W.shape), dtype=float)
        return self._add_padding(X), Y

    @abstractmethod
    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        pass

    @abstractmethod
    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> \
            Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        pass


class FourForLoops(Conv2DAlgo):
    """Najjednostavniji algoritam. Kod prolaska unapred imamo 4 for petlje kojima redom, za svaki primer unutar batch-a,
    za svaku izlaznu mapu karakteristika, za svaki piksel, računamo aktivacioni potencijal.
    Kod prolaska unazad istim redosledom sa 4 for petlje prolazimo kroz izlazni tenzor i računamo potrebne parcijalne izvode."""

    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        X, Y = self.prepare(X, W)

        output_h, output_w = Y.shape[2], Y.shape[3]
        kernel_h, kernel_w = W.shape[2], W.shape[3]
        for n in range(X.shape[0]):
            for out_map_i in range(W.shape[0]):
                for i in range(output_h):
                    for j in range(output_w):
                        # kada znamo (i, j) za izlaznu mapu karakteristika treba da pronađemo "parče" ulaznih mapa karakteristika
                        # na koje treba primeniti filtere da bismo dobili baš taj izlaz na mestu (i, j)
                        # parče će biti u granicama [input_i_start, input_i_end) po visini i [input_j_start, input_j_end) po širini.
                        in_i_start = i * self.s
                        in_j_start = j * self.s

                        in_i_end = in_i_start + kernel_h
                        in_j_end = in_j_start + kernel_w

                        Y[n, out_map_i, i, j] = \
                            xp.sum(X[n, :, in_i_start: in_i_end,
                                   in_j_start: in_j_end] * W[out_map_i]) + b[out_map_i]

        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        dEdX = xp.zeros_like(X)
        dEdW = xp.zeros_like(W)
        dEdb = xp.zeros((dEdO.shape[-3], ), dtype=float)

        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        for n in range(X.shape[0]):
            for out_map_i in range(W.shape[0]):
                dEdb[out_map_i] += xp.sum(dEdO[n, out_map_i, :, :])
                for i in range(output_h):
                    for j in range(output_w):
                        # ista priča sa indeksima kao i kod prolaska unapred.
                        # samo što ćemo sada da gledamo obrnuto, iz ugla "ko na koga utiče".
                        # Računajući izlaz na mestu nb, out_map_idx, i, j,
                        # koje je sve težine to parče ulazne mape karakteristika koristilo. Tj. gledamo uticaje
                        # težina na izlaze, kao i uticaje ulaza na izlaze.
                        # Kod računanja parcijalnih izvoda po pomerajima situacija je ista, s tim što je još lakše.
                        # prosto, i-ti pomeraj utiče na celu i-tu izlaznu mapu karakteristika za svaki primer u batch-a
                        in_i_start = i * self.s
                        in_j_start = j * self.s

                        in_i_end = in_i_start + W.shape[2]
                        in_j_end = in_j_start + W.shape[3]

                        dEdX[n, :, in_i_start: in_i_end, in_j_start: in_j_end] += \
                            dEdO[n, out_map_i, i, j] * W[out_map_i]

                        dEdW[out_map_i] += dEdO[n, out_map_i, i, j] * X[n, :, in_i_start: in_i_end, in_j_start: in_j_end]

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb


class ThreeForLoops(Conv2DAlgo):
    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray,
                 b: xp.ndarray) -> xp.ndarray:
        X, Y = self.prepare(X, W)

        # U profesorovim prezentacijama data je implementacija prolaska unapred sa 2 i 4 for petlje.
        # Možemo da odradimo potrebnu računicu i sa 3.
        # Za svaku izlaznu mapu karakteristika krećemo se po svakom pikselu, ali za sve primere u batch-u istovremeno.
        output_h, output_w = Y.shape[2], Y.shape[3]
        kernel_h, kernel_w = W.shape[2], W.shape[3]

        for out_map_i in range(W.shape[0]):
            for i in range(output_h):
                for j in range(output_w):
                    # kada znamo (i, j) za izlaznu mapu karakteristika treba da pronađemo "parče" ulaznih mapa karakteristika
                    # na koje treba primeniti filtere da bismo dobili baš taj izlaz na mestu (i, j)
                    # parče će biti u granicama [input_i_start, input_i_end) po visini i [input_j_start, input_j_end) po širini.
                    in_i_start = i * self.s
                    in_j_start = j * self.s

                    in_i_end = in_i_start + kernel_h
                    in_j_end = in_j_start + kernel_w

                    # Razmotrimo narednu liniju koda.
                    # Iz ndimenzionalnog niza padded_input uzimamo 4D "parče" takvo da uzimamo sve elemente po prve dve dimenzije,
                    # tj. za svaki primer u batch-u i svaku ulaznu mapu karakteristika i elemente po poslednje dve dimenzije u skladu sa
                    # izračunatim vrednostima input_i_start, input_i_end, input_j_start, input_j_end.
                    # Deo niza padded_input koji ćemo na taj način uzeti je dimenzija Nb x D_l x k_dim x k_dim,
                    # gde je Nb broj primera u batch-u, D_l broj ulatnih mapa karakteristika, a k_dim veličina filtera.
                    # Takvo "parče" 4D tenzora množimo sa self._W[out_map_idx]
                    # self._W[out_map_idx] je 3D tenzor filtera koji se primenjuju za izlaznu mapu karakteristika out_map_idx.
                    # Taj 3D tenzor je dimenzija D_l x k_dim x k_dim.
                    # Dakle, hoćemo da pomnožimo nešto dimenzije Nb x D_l x k_dim x k_dim sa nečim dimenzija D_l x k_dim x k_dim.
                    # Ono što će numpy uraditi sa takvim zahtevom je broadcast 3D tenzora, tj. kao da ćemo zapravo množiti sa
                    # nečim što je dimenzije Nb x D_l x k_dim x k_dim, kao da će se sadržaj 3D tenzora iskopirati Nb-1 puta tako da
                    # popuni tenzor dimenzije Nb x D_l x k_dim x k_dim.
                    # Takav proizvod nam odgovara. Dakle, radimo automatski za svaki primer u batch-u.
                    # Kada budemo imali takav proizvod (koji je naravno dimenzija Nb x D_l x k_dim x k_dim), možemo dobiti aktivacione
                    # potencijale za datu izlaznu mapu out_map_idx na mestu i, j za sve primere u batch-u.
                    # Potrebno je samo da taj rezultat sumiramo po svim dimenzijama osim po prvoj,
                    # jer po prvoj su nam primeri u batch-u.
                    # To postižemo funkcijom xp.sum, a dimenzije po kojima sumiramo su (1, 2, 3), tj. sve osim prve
                    xp.sum(X[:, :, in_i_start: in_i_end, in_j_start: in_j_end] * W[out_map_i],
                           axis=(1, 2, 3),
                           out=Y[:, out_map_i, i, j])

                    # U prethodnom objašnjenju rečeno je da prethodna linija računa potencijale.
                    # Da ne bismo komplikovali već opširno objašnjenje to je bio dovoljno dobar izraz,
                    # iako potencijale nismo pomoću te linije u potpunosti izračunali.
                    # Nedostaje nam samo još da dodamo pomeraj što u sledećoj liniji činimo.
                    # Naravno, dodajemo pomeraj za svaki primer u batch-u. I ovde imamo broadcasting jer sabiramo
                    # niz dimenzija Nb x 1 i skalar (self._b[out_map_idx] je skalar)
                    xp.add(Y[:, out_map_i, i, j], b[out_map_i], out=Y[:, out_map_i, i, j])

        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        dEdX = xp.zeros_like(X)
        dEdW = xp.zeros_like(W)
        dEdb = xp.zeros((dEdO.shape[-3],), dtype=float)

        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        # Rešenje sa 3 petlje neće biti tako trivijalno kod prolaska unazad. Problem je što nećemo moći da se
        # oslonimo na broadcasting kod množenja dEdO[:, out_map_idx, i, j] * self._W[out_map_idx], jer
        # su dimenzije ovih nizova Nb i D x k x k. Ono što bismo voleli je da se postigne broadcast koji bi naš 3D tenzor
        # "ponovio" Nb puta.
        # Da postignemo željeno ponašanje, možemo koristiti einsum funkciju.
        # Korišćenjem 3 petlje umesto 4 kod prolaska unazad postiže se 10 puta brži kod.

        for out_map_idx in range(W.shape[0]):
            dEdb[out_map_idx] = xp.sum(dEdO[:, out_map_idx, :, :])
            for i in range(output_h):
                for j in range(output_w):
                    in_i_start = i * self.s
                    in_j_start = j * self.s

                    in_i_end = in_i_start + W.shape[2]
                    in_j_end = in_j_start + W.shape[3]

                    # xp.einsum("i,jkl->ijkl", dEdO[:, out_map_idx, i, j], kernel_tensor[out_map_idx])
                    # očekujemo 4D izlaz Nb x D x k x k
                    # na mestu i, j, k, l izlaza treba da se nađe proizvod i-tog elementa iz prvog niza i
                    # elementa na mestu j, k, l drugog niza. Dakle, navođenjem "i,jkl->ijkl" postižemo upravo to što želimo.
                    dEdX[:, :, in_i_start: in_i_end, in_j_start: in_j_end] += \
                        xp.einsum("i,jkl->ijkl", dEdO[:, out_map_idx, i, j], W[out_map_idx])

                    # xp.einsum("i,ijkl->jkl", dEdO[:, out_map_idx, i, j], self.padded_input[:, :, input_i_start: input_i_end,
                    # input_j_start: input_j_end]
                    # Kod računanja izvoda greške po matrici težina, trebalo bi da saberemo parcijalne izvode za sve primere u batch-u.
                    # i-ti element prvog niza treba pomnožiti sa i-tim 3D tenzorom drugog 4D niza, a na kraju treba sabrati po 0 dimenziji.
                    # Izraz "i,ijkl->jkl" radi upravo to. Dobijamo 3D izlaz (3 indeksa nakon ->).
                    # Indeks i nalazi se samo sa leve strane, ne i desne što znači da treba sumirati tako da se ta dimenzija "izgubi".
                    dEdW[out_map_idx] += xp.einsum("i,ijkl->jkl", dEdO[:, out_map_idx, i, j],
                                                   X[:, :, in_i_start: in_i_end, in_j_start: in_j_end])

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb


class TwoForLoops(Conv2DAlgo):
    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        X, Y = self.prepare(X, W)

        # Ovde je data implementacija prolaska unapred koja koristi dve for petlje.
        # Za svaku izlaznu mapu karakteristika i svaki primer u batch-u računaćemo istovremeno
        # aktivacioni potencijal piksel po piksel

        output_h, output_w = Y.shape[2], Y.shape[3]
        kernel_h, kernel_w = W.shape[2], W.shape[3]

        for i in range(output_h):
            for j in range(output_w):
                # priča sa indeksima je ista kao i ranije...
                in_i_start = i * self.s
                in_j_start = j * self.s

                in_i_end = in_i_start + kernel_h
                in_j_end = in_j_start + kernel_w

                # Sada treba da dva 4D "parčeta" izmnožimo tako da dobijemo 2D parče. Proizvod treba biti pokomponentan,
                # a rezultate u poslednje dve dimenzije treba sažeti u jedan broj, tj. sabrati (poslednje dve dimenzije
                # su nam za parče neke ulazne mape karakteristika nekog primera u batch-u u padded_input, odnosno za
                # matricu filtera u 4D tenzoru filtera.)
                # U rezultatu einsum operacije koji je 2D na mestu (i, m) treba da se nađe rezultat koji se odnosi na i-ti primer
                # u batch-u i m-tu izlaznu mapu karakteristika. A vrednost na tom mestu dobijamo tako što pokomponentno
                # pomnožimo po indeksima j, k, l i na kraju dimenzije koje su odgovarale indeksima j, k, l sažmemo tako što ih saberemo.
                Y[:, :, i, j] = xp.einsum('ijkl,mjkl->im', X[:, :, in_i_start: in_i_end, in_j_start: in_j_end], W)
                Y[:, :, i, j] += b

        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        dEdX = xp.zeros_like(X)
        dEdW = xp.zeros_like(W)
        dEdb = xp.zeros((dEdO.shape[-3],), dtype=float)

        output_w = dEdO.shape[-1]
        output_h = dEdO.shape[-2]

        # Rešenje sa dve petlje (pomoću einsum) bi bilo:
        for i in range(output_h):
            for j in range(output_w):
                in_i_start = i * self.s
                in_j_start = j * self.s

                in_i_end = in_i_start + W.shape[2]
                in_j_end = in_j_start + W.shape[3]

                dEdX[:, :, in_i_start: in_i_end, in_j_start: in_j_end] += \
                    xp.einsum("ij,jklm->iklm", dEdO[:, :, i, j], W)

                dEdW += xp.einsum("ij,iklm->jklm", dEdO[:, :, i, j], X[:, :, in_i_start: in_i_end,
                                  in_j_start: in_j_end])
                dEdb += xp.sum(dEdO[:, :, i, j], axis=0)

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb


class Matmul(Conv2DAlgo):
    """
    Ideja ovog algoritma (i narednog) je dosta drugačija. Sada ciljamo na to da željene rezultate dobijemo
    matričnim proizvodima, ali ne pomoću velikog broja njih kao ranije, već samo jednog (po primeru) kod prolaska unapred i
    dva kod prolaska unazad. Željeno ponašanje ne može se ostvariti množenjem sa matricama koje dobijemo kao ulazne
    podatke, već moramo sami napraviti "pogodne" matrice takve da će nakon proizvoda rezultat biti upravo to što želimo.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, X: xp.ndarray, W: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
        Y_shape = self._output_shape(X.shape, W.shape)
        X = self._add_padding(X)

        # potrebno je da prvo dobijemo "specijalnu" matricu transformacijom ulaza takvu da množenjem kernela (koji će takođe biti matirca)
        # sa njom dobijemo baš rezultat koji nam je potreban.
        # Za dobijanje takve matrice koristimo funkciju im2col

        Xcol = im2col(X, W.shape[-2], W.shape[-1], self.s)

        # Potrebno je od kernela dobiti matricu dimenzija Dl+1 x (Dl * Kh * Kw)
        Wmat = W.reshape((W.shape[0], -1))  # reshape je brz, jer se sadrzaj memorije ne premesta

        # Rezultat funkcije im2col je 3D tenzor dimenzija Nb x (Dl * Kh * Kw) x (Oh * Ow), gde su Oh i Ow visina i širina svake
        # izlazne mape karakteristika.
        # Proizvod dva ndimenzionalna niza formata Dl+1 x (Dl * Kh * Kw) i Nb x (Dl * Kh * Kw) x (Oh * Ow) pomoću matmul
        # kao rezultat daje ndimenzionalni niz formata Nb x Dl+1 x (Oh * Ow), jer matmul onda kada dobije 3D tenzor njega smatra
        # nizom matrica i radi po jedan matrični proizvod za svaku od matrica i kao rezultat dobijemo isto toliko matrica, tj.
        # opet 3D tenzor. A s obzirom na to da je 3D tenzor im2col pažljivo konstruiran, rezultat množenja (koji je formata
        # Nb x Dl+1 x (Oh * Ow)) je već "gotov" rezultat, potrebno je samo da ga proglasimo za 4D tenzor i razdvojimo poslednju
        # dimenziju na dve. Nakon svega toga dodaćemo i pomeraj.

        Y = xp.matmul(Wmat, Xcol)
        Y = Y.reshape(Y_shape)

        # ostaje da dodamo pomeraj...
        for o_ch in range(Y.shape[1]):
            Y[:, o_ch, :, :] += b[o_ch]

        return Y

    def backward(self, dEdO: xp.ndarray, X: xp.ndarray, W: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        X = self._add_padding(X)
        Xcol = im2col(X, W.shape[-2], W.shape[-1], self.s)
        dEdb = xp.zeros((dEdO.shape[-3],), dtype=float)

        xp.sum(dEdO, axis=(0, 2, 3), out=dEdb)  # parcijalne izvode po pomeraju lako računamo...

        WT = (W.reshape((W.shape[0], -1))).T  # prvo kao u prolasku unapred pravimo linearizovanu matricu,
        # a zatim je transponujemo.

        dAmat = dEdO.reshape((dEdO.shape[0], dEdO.shape[1], -1))  # parcijalne izvode po izlazu takođe pretvaramo u matricu,
        # tj. niz matrica (po jedna za svaki primer u batch-u)

        # Sada treba da pomnožimo matricu WT i matrice dAmat (3D tenzor)
        dXcol = xp.matmul(WT, dAmat)

        # dXcol, rezultat množenja WT i dAmat, NIJE "gotova" matrica parcijalnih izvoda po ulazima.
        # dXcol je dimenzija Nb x (Dl * Kh * Kw) x (Oh * Ow), a tenzor parcijalnih izvoda dEdX je dimenzija
        # Nb x Dl x Ih x Iw, gde su Ih i Iw visina i širina ulaznih mapa karakteristika.
        # dXcol sadrži "delove" parcijalnih izvode koje treba sabrati i smestiti u dEdX, što radi funkcija col2im.
        # Više o tome šta se nalazi u matrici dXcol nalazi se u okviru same funkcije col2im
        dEdX = col2im(dXcol, X.shape, W.shape[-2], W.shape[-1], self.s)

        # Računanje parcijalnih izovda po težinama je jednostavnije.
        # Treba samo pomnožiti matricu parcijalnih izvoda sa transponovanom matricom dobijenom sa im2col iz prolaska unapred
        # i rezultat množenja sabrati za svaki primer u batch-u.
        dEdW = xp.einsum('ijl,ikl->jk', dAmat, Xcol)
        dEdW = dEdW.reshape(W.shape)

        dEdX = self._remove_padding(dEdX)

        return dEdX, dEdW, dEdb
