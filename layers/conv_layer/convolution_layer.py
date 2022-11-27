from models.adaptive_object import AdaptiveObject
from layers.conv_layer.conv_layer_algorithms import *
from weight_initializers.random_initialize import rand_init


class Convolution2D(AdaptiveObject):
    """ Konvolucioni sloj primenjuje se najčešće na ulaze koji su trodimenzionalni.
        Ulaz konvolucionog sloja čini veći broj mapa karakteristika koje su dvodimenzionalne.
        Kod prvog konvolucionog sloja u pitanju su najčešće slike koje su zapravo sačinjene od
        3 kanala (red, green, blue), tj. tri mape karakteristika, a svaka mapa karakteristika je matrica.

        Na ulaze primenjujemo veći broj filtara i tako dobijamo izlazne aktivacione potencijale koji su, ponovo,
        3D, tj. sastoje se iz većeg broja mapa karakteristika.

        Za dobijanje i-tog 3D tenzora aktivacionih potencijala za i-tu izlaznu mapu karakteristika potrebno je da
        na svaku ulaznu mapu karakteristika primenimo po jedan 2D filter (kernel) i "iskombinujemo" (saberemo) dobijene
        rezultate. Na kraju dodajemo i pomeraj za i-tu izlaznu mapu karakteristika.

        Za svaki par ulazna-izlazna mapa karakteristika imamo zaseban filter, pa je ukupan broj 2D filtara jednak
        zeljeni_broj_izlaznih_mapa_karakteristika x br_ulaznih_mapa_karakteristika
        S obzirom na to, parametri filtera čine 4D tenzor dimenzija:
        zeljeni_broj_izlaznih_mapa_karakteristika x br_ulaznih_mapa_karakteristika x velicina_filtera x velicina_filtera,
        pri čemu smo pretpostavili da su filteri kvadratni (mada ne komplikuje se implementacija ni ako dozvolimo uopštenje)
    """

    @staticmethod
    def get_height(input_height: int, kernel_height: int, padding: int, stride: int) -> int:
        """Znajući visinu ulaza, visinu kernela (fitlera), padding i stride, možemo izračunati visinu izlaza"""
        return (input_height - kernel_height + 2 * padding) // stride + 1

    @staticmethod
    def get_width(input_width: int, kernel_width: int, padding: int, stride: int) -> int:
        """Znajući širinu ulaza, širinu kernela (fitlera), padding i stride, možemo izračunati širinu izlaza"""
        return (input_width - kernel_width + 2 * padding) // stride + 1

    def __init__(self, in_maps_n: int, out_maps_n: int, kernel_size: int = 7,
                 padding: int = 0, stride: int = 1, algorithm: Conv2DAlgo = FourForLoops(),
                 name: str = 'Convolutional Layer'):
        super().__init__(name)
        self.in_maps_n = in_maps_n
        self.out_maps_n = out_maps_n

        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

        w = self.in_maps_n * kernel_size * kernel_size
        W_shape = (out_maps_n, self.in_maps_n, kernel_size, kernel_size)
        self._W = rand_init(out_maps_n, w).reshape(W_shape)
        self._b = xp.zeros((out_maps_n,))

        self._dEdW = None
        self._dEdb = None

        # S obzirom na to da možemo računati prolaske unapred i unazad na više načina, zgodno je da tu logiku prebacimo u zasebnnu
        # hijerarhiju klasa. Kada nam bude bio potreban određeni rezultat, mi objektu koji "ume" da izračuna to što nam treba
        # prosleđujemo zahtev i on nam daje nazad rezultate. U strožije tipiziranim jezicima taj objekat bi trebalo da bude
        # tipa takvog da su iz njega izvedeni svi konkretni algoritmi za naš problem. Ovde je takva klasa nazvana
        # ConvolutionLayerAlgorithm i apstraktna je (ne može se kreirati objekat te klase, već služi samo za nasleđivanje)
        # Svaki objekat te ili izvedene klase može da izračuna prolazak unapred i unazad kod konvolucionog sloja.
        self.algo = algorithm
        self.algo.pad = self.padding
        self.algo.s = self.stride

    def __call__(self, X: xp.ndarray) -> xp.ndarray:
        return self.algo(X, self._W, self._b)

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        dEdX, self._dEdW, self._dEdb = self.algo.backward(dEdO, self._inputs, self._W)

        return dEdX

    @property
    def parameters(self) -> tuple:
        return self._W, self._b

    @parameters.setter
    def parameters(self, val: tuple):
        self._W, self._b = val

    def update_parameters(self):
        self._optimizer.update_parameters(self._W, self._dEdW)
        self._optimizer.update_parameters(self._b, self._dEdb)
        self._dEdW = None
        self._dEdb = None
