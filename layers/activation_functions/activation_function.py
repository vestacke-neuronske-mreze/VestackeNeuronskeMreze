from abc import abstractmethod

from backend.backend import xp
from layers.function import Function


class ActivationFunction(Function):
    """Aktivaciona funkcija mora biti u stanju da se primeni na ulazne podatke i moramo
      biti u stanju da na osnovu ulaznih podataka dobijemo izvod funkcije.

      Za sve aktivacione funkcije osim Softmax funkcije važi da i-ti izlaz zavisi samo od i-tog ulaza.
      U opštem slučaju, output funckije derivative trebalo bi da bude Nb x n x n, gde je Nb broj primera u batch-u,
      a n broj parametara svakog od primera. Taj 3D tenzor treba tretirati kao niz (dužine Nb) matrica parcijalnih izvoda.
      Pod gore navedenom pretpostavkom, svaka od matrica biće dijagonalna (osim kod Softmax funkcije).
      Kod izračunavanja parcijalnih izvoda kod propagiranja greške unazad,
      parcijalne izvode greške po izlazu iz aktivacionog sloja treba pomožiti sa ovim parcijalnim izvodima.
      Ukoliko množimo "naslagane" redove parcijalnih izvoda greške po izlazu iz aktivacionog sloja sa
      "naslaganim" matricama parcijalnih izvoda aktivacione funkcije po ulazima, a te matrice su dijagonalne,
      onda se određeni proizvodi red * matrica svodi na pokomponentni prozivod tog reda i reda koji je dobijen
      tako što smo iz dijagonalne matrice izvukli dijagonalu i poređali ih u red.
      Ta činjenica se u implementacijama često koristi (pa i ovde), tako da će funkcija derivative
      vraćati rezultat koji je istog "oblika" (~shape) kao i ulaz (a kasnije nam ostaje da samo
      odradimo pokomponentni proizovd)
      """

    def __init__(self, name: str = None):
        super().__init__(name=name)

    @abstractmethod
    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        pass

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        from layers.activation_functions.softmax import Softmax

        if isinstance(self, Softmax):
            dEdO = dEdO.reshape((dEdO.shape[0], 1, dEdO.shape[-1]))
            dEdI = xp.matmul(dEdO, self.deriv(self._inputs))
            return dEdI.reshape((dEdI.shape[0], dEdI.shape[-1]))

        return xp.multiply(dEdO, self.deriv(self._inputs))

