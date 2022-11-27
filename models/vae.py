from typing import Tuple

from backend.backend import xp
from models.feedforward_nn import Model
from loss_functions.abstract_loss_function import LossFunction
from loss_functions.kl_divergence import DKLStandardNormal
from optimizers.abstract_optimizer import Optimizer


class VAE(Model):
    """S obzirom na funkcionalnosti koje bi varijacioni autoenkoder trebalo da ima (da može da se trenira, propagira signal unapred,
        grešku unazad itd), klasu ćemo izvesti iz klase Network.

        Varijacioni autoenkoder sastojaće se iz dve mreže. Zadatak prve biće da na osnovu ulaznih podataka generiše
        vektore mi i gama uz pomoć kojih možemo dobiti latentnu slučajnu promenljivu z kao z = mi + sqrt(exp(gamma)) * eps, gde
         eps~N(0, 1).
        Enkoder će kao kriterijumsku funkciju imati Kulbak Lajberovo rastojanje normalne raspodele sa parametrima (mi, exp(gama)) i
        normalne raspodele sa parametrima (0, 1).
        Ulaz dekodera je skrivena slučajna promenljiva z = mi + sqrt(exp(gamma)) * eps, a cilj je da dekoder na osnovu tog ulaza
        generiše izlaz "što sličniji" ulazu koji smo prosledili enkoderu.
        Posao dekodera se dakle svodi na sve što je ranije već viđeno: imamo vrednosti koje "gađamo" i naša mreža treba da nauči
        da ih pogađa. Formalno, cilj je maksimizirati verodostojnost, a konkretna kriterijumska funkcija dobija se na osnovu
        pretpostavke o raspodeli p(X/z). Kao i ranije, svodi se na neku od entropija ili Euklidovo rastojanje (MSE)

        Kod prolaska unazad, dekoder će vratiti parcijalne izvode dE2/dz, gde je sa E2 označena greška dekodera.
        Poslednji sloj enkodera treba da generiše gama i mi. Promena težina u poslednjem sloju enkodera utiče na grešku
        koju enkoder pravi (zvaćemo je E1), ali i na ulaz dekodera i samim tim grešku koju dekoder pravi.
        S obzirom na to, kod prolaska unazad i računanja parcijalnih izvoda ukupne greške po parametrima poslednjeg sloja
        enkodera treba uzeti u obzir (dodati) i parcijalne izvode dE2/dmi i dE2/dgama koje lako dobijamo kao:
            dE2/dmi = dz/dmi * dE2/dz,
            dE2/dgama = dz/dgamma * dE2/dz.

        Praktična teškoća sa implementacijom opisanog ponašanja sastoji se u tom kombinovanju parcijalnih izvoda iz dve mreže, ali
        i u sledećem:
        Klasa Network implementirana je tako da ne podržava "račvanje" tj. to da u nekom sloju generišemo zasebnim parametrima dva
        izlaza ili da nakon nekog sloja na istom nivou imamo dva ili više sloja. Naš model je sekvencjialni. Kriterijumske funkcije
        su takođe implementirane tako da dobijaju jedan ulaz i vraćaju jedan izlaz.

        Kao i uvek, pokušavamo da od postojećeg koda bez ikakvih izmena izvučemo što više možemo. Jedan od trikova koje možemo
        upotrebiti ovde a koji ne zahteva nikakve promene postojećeg koda je da mi i gama generišemo istim gusto povezanim slojem
        kao vektor dužine 2*D, gde je D dimenzionalnost skrivene promenljive z.

        Tako bi poslednji sloj (neračunajući kriterijumsku funkciju) enkodera bio običan Dense sloj. Njegov je zadatak da generiše
        neke izlaze dužine 2*D i da svoje parametre ažurira koristeći parcijalne izvode greške po izlazima iz sloja. Jedino o čemu
        treba da vodimo računa u vezi sa poslednjim slojem enkodera je da kod prolaska unazad prosledimo odgovarajuće parcijalne
        izvode. Ostatak mreže će raditi svoj posao kao i ranije.

        Dakle, ovim naš problem izmeštamo, ali ga ne rešavamo. Ali van mreže (neračunajući kriterijumsku funkciju kao
        poslednji sloj mreže) taj problem je lako rešiv.
        Kriterijumska funkcija DKL će dobijati ulaze veličine 2*D koje ćemo tretirati kao da je prva polovina mi, a druga gama.
        Isto tako radićemo i prolazak unazad i izračunavanje parcijalnih izvoda. Parcijalni izvodi će biti veličine 2D
        i prvu polovinu parcijalnih izvoda računamo kao dE1/dmi, a drugu kao dE1/dgama.

        Tako dobijene parcijalne izvode treba sabrati (naravno, van kriterijumske funkcije) sa parcijalnim izvodima po istim tim
        parametrima po grešci dekodera i tako dobijene parcijalne izvode veličine 2*D treba proslediti za ostatak propagacije
        greške unazad kroz enkoder (dakle, prosleđujemo ih poslednjem sloju dekodera itd).
    """

    class VAELoss(LossFunction):

        def __init__(self, decoder_loss: LossFunction):
            super().__init__("VAE loss")
            if decoder_loss is None:
                raise Exception("Decoder loss function cannot be None!")
            self.encoder_loss = DKLStandardNormal()
            self.decoder_loss = decoder_loss
            self.mu: xp.ndarray = None
            self.gamma: xp.ndarray = None

        def __call__(self, y: xp.ndarray, t: xp.ndarray) -> float:
            el = self.encoder_loss(self.mu, self.gamma)
            dl = self.decoder_loss(y, t)

            return el + dl

        def backward(self, y: xp.ndarray, t: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
            d_mu, d_gamma = self.encoder_loss.backward(self.mu, self.gamma)
            d_x = self.decoder_loss.backward(y, t)
            return d_mu, d_gamma, d_x

    def __init__(self, encoder: Model, decoder: Model, m: int, name="VAE"):
        super().__init__(name=name)

        self.encoder: Model = encoder
        self.decoder: Model = decoder

        self._loss = VAE.VAELoss(self.decoder._loss)

        self.m: int = m
        self.z: xp.ndarray = None
        self.eps: xp.ndarray = None
        self.gamma: xp.ndarray = None

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        self.encoder.set_optimizer(optimizer)
        self.decoder.set_optimizer(optimizer)

    def update_parameters(self):
        self.encoder.update_parameters()
        self.decoder.update_parameters()

    @property
    def parameters(self) -> list:
        return [self.encoder.parameters,
                self.decoder.parameters]

    @parameters.setter
    def parameters(self, val: tuple):
        self.encoder.parameters, self.decoder.parameters = val

    def _get_z(self, mu_gamma: xp.ndarray) -> xp.ndarray:
        Nb = mu_gamma.shape[0]
        eps = xp.random.normal(size=(Nb, self.m))

        mu = mu_gamma[:, :self.m]
        gamma = mu_gamma[:, self.m:]

        self._loss.mu = mu
        self._loss.gamma = gamma

        z = eps * xp.exp(gamma * 0.5) + mu

        if self.training:
            self.eps = eps
            self.z = z
            self.gamma = gamma

        return z

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        tmp = self.encoder(inputs)
        self.z = self._get_z(tmp)
        return self.decoder(self.z)

    def backward(self, dE: xp.ndarray) -> xp.ndarray:
        d_mu, d_gamma, dx = dE

        dz = self.decoder.backward(dx)
        d_mu += dz
        d_gamma += dz * 0.5 * xp.exp(0.5 * self.gamma) * self.eps
        dmuGamma = xp.hstack((d_mu, d_gamma))

        return self.encoder.backward(dmuGamma)

    def generate_new_samples(self, num_of_samples: int = 1, samples_like: xp.ndarray = None) -> xp.ndarray:
        """Cilj generativnog autoenkodera je da generiše nove uzorke. Može ih generisati po uzoru na neke
        uzorke ili potpuno nasumično.

        Ukoliko generišemo na osnovu uzoraka, onda treba da ponovimo isto ono što radimo i kod prolaska unapred.
        A ukoliko generišemo potpuno nasumično, onda samo treba da propagiramo unapred signal kroz dekoder z koje dobijamo
        iz normalne normirane raspodele.

        To je suština ove metode. Ostalo su samo implementacioni detalji koji nisu od značaja za razumevanje varijacionog
        autoenkodera.
        """

        if samples_like is not None:
            return self(samples_like)

        z = xp.random.normal(size=(num_of_samples, self.m))
        return self.decoder(z)

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        self.encoder.training = val
        self.decoder.training = val

