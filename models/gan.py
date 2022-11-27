from typing import List

from backend.backend import xp
from models.feedforward_nn import Model
from metrics.metrics import Metric
from optimizers.abstract_optimizer import Optimizer
from utils.dataset import Dataset


class GAN(Model):
    """
    Generativne protivničke mreže (Generative Adversarial Networks - GAN, engl) treba da budu u mogućnosti
    da se treniraju, da propagiraju signal unapred i parcijalne izvode unazad, da ažuriraju parametre itd.
    Potrebno je da imaju skoro sve funkcionalnosti koje ima i klasa Network, pa je i ova klasa, poput VAE klase,
    izvedena iz klase Network.

    Potrebne su nam dve mreže. Mreža generator i mreža diskriminator. Zadatak mreže generatora je da
    na osnovu slučajnog šuma generiše izlaz koji je "sličan" uzorcima iz skupa uzoraka za treniranje.
    Zadatak diskriminatora je da nauči da razlikuje uzorke koji su "pravi", tj. potiču iz raspodele uzoraka
    za treniranje i one koje dolaze iz rasapodele generatora, tj. kao izlaz generatora.

    U pogledu implementacije, mreža diskriminator davaće na izlazu verovatnoću da je uzorak koji je dobila
    na ulazu iz skupa uzoraka za treniranje, a ne izlaz generatora. Dakle, dimenzija izlaza je 1, a kao
    kriterijumsku funkciju koristimo binarnu entropiju.

    Mreža generator davaće izlaz istog formata kog su i uzorci iz skupa uzoraka za treniranje. Ulaz je nasumični šum.
    Generator neće imati kriterijumsku funkciju jer će izlaz generatora biti ulaz diskriminatora, pa putem mreže
    diskriminatora možemo da treniramo i generator (od diskriminatora dobijamo parcijalne izvode).

    Kod treniranja generativnih protivničkih mreža ne treniramo obe mreže odjednom, već ih treniramo naizmenično.
    Ukoliko treniramo diskriminator, potrebno je da diskriminatoru na ulazu damo uzorke iz obe klase ("prave" i "lažne").
    "Lažne" uzorke dobijamo tako što nasumični šum propustimo kroz generator. Labela "lažnih" uzoraka je 0, a "pravih" je 1.
    Da bi implementacija bila lakša, diskriminator treniramo sa dva batch-a podataka (iz skupa uzoraka i iz generatora).
    Kod batch-a sa "pravim" uzorcima, uzorke saljemo direktno diskriminatoru, a kod batch-a sa "lažnim", prvo treba da te lažne
    uzorke dobijemo, tj. da šum propustimo kroz generator.

    Kod treniranja generatora, na ulazu imamo slučajni šum i cilj je da generator šum transformiše u izlaz istog formata
    kao i podaci iz skupa uzoraka za treniranje, a da uz to dobijeni izlaz bude dovoljno sličan podacima iz ulaznog skupa
    da diskriminator takve ulaze klasifikuje kao "prave". Gledano iz ugla optimizacije i generatora, prilikom treniranja
    generatora ciljna labela generisanih uzoraka je 1.
    """

    def __init__(self, generator: Model, discriminator: Model, z_size: int, k: int = 1):
        super().__init__()

        self.generator: Model = generator
        self.discriminator: Model = discriminator
        self.z_size = z_size
        self.k = k

        self.train_generator = True
        self.noise = False

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        self.generator.set_optimizer(optimizer)
        self.discriminator.set_optimizer(optimizer)

    def _epoch(self, data: Dataset,
               metrics: List[Metric] = []) -> float:
        loss = 0.0
        batch_num = 0

        for batch_x, _ in data:

            batch_num += 1
            nb = len(batch_x)
            x_fake = self.generate_new_samples(nb)
            y_0 = xp.zeros((nb, 1), dtype=float)
            y_1 = xp.ones((nb, 1), dtype=float)
            x = xp.vstack((x_fake, batch_x))
            y = xp.vstack((y_0, y_1))

            output, _, l = self.discriminator._process_minibatch(x, y)
            self.discriminator.update_parameters()
            loss += l  # funkciju greške prikazivaćemo kao prosečnu po svakom primeru

            for m in metrics:
                m.calculate(output, y)

            if batch_num % self.k == 0:
                x_fake = self.generate_new_samples(nb)
                _, dEdG, _ = self.discriminator._process_minibatch(x_fake, y_1)
                self.generator.backward(dEdG)
                self.generator.update_parameters()

        loss /= batch_num
        for m in metrics:
            m.calculate_for_epoch()
        return loss

    def generate_new_samples(self, num_of_samples: int = 1) -> xp.ndarray:
        """Kada istreniramo GAN ovom funkcijom generišemo nove primere."""
        z = xp.random.normal(0, 1, (num_of_samples, self.z_size))
        return self.generator(z)

    @property
    def parameters(self) -> list:
        return [self.generator.parameters,
                self.discriminator.parameters]

    @parameters.setter
    def parameters(self, val: tuple):
        self.generator.parameters, self.discriminator.parameters = val

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        self.generator.training = val
        self.discriminator.training = val
