from backend.backend import xp
from abc import ABC, abstractmethod


class Function(ABC):
    """Ova apstraktna klasa predstavlja jednu od najbitnijih klasa i veliki broj klasa biće izveden iz nje
        (slojevi neuronske mreže, kriterijumske funkcije itd). Zadatak funkcije je da za neke ulaze proizvede izlaze,
        ali je takođe bitno da obezbedimo funkcionalnosti za propagaciju greške unazad.

        Konkretnije, funkcija će raditi sa ndimenzionalnim nizovima - tenzorima.
        Zbog performansi, izbegavaćemo stalno alociranje i dealociranje memorije, uz optimističnu pretpostavku da
        na raspolaganju imamo dovoljno memorije. U realnijoj situaciji, trebalo bi implementirati ceo podsistem za
        upravljanje memorijom sa memorijskim "pulom", ali ovde nam neće biti fokus na tim detaljima.

        Cena toga što ćemo memorju zauzimati samo jednom je malo složeniji kod. Naime, izbegavaćemo i alokaciju privremene
        memorije za međurezultate tako da ćemo često koristiti out parametar kod numpy/cupy funkcija. Da bismo koristili out
        argument često je potrebno da imamo i pomoćni niz (obično se odgovarajuće promenljive nazivaju tmp_array i sl.)
        Takođe želimo da od korisnika sakrijemo implementacione detalje poput veličine mini batch-a, kao i da ga gde god je to
        moguće oslobodimo čak i obaveze da u konkretnom sloju navede broj ulaznih/izlaznih neurona - videti neki od primera
        iz foldera text_examples.

        NAPOMENA: opisane fleksibilnosti i optimizacije NISU od ključnog značaja za razumevanje šire slike
        arhitekture klasa, organizacije koda i sl. Ovakve "sitnice" međutim jesu primer problema sa kojima
        se suočavamo prilikom implementacije.

        Bavićemo se neuronskim mrežama linearne arhitekture, što podrazumeva da su slojevi prosto sekvencijalno "poređani", tj.
        da svaki sloj ima tačno jedan ulaz, tačno jedan izlaz i da izlaz jednog sloja može biti ulaz u tačno jedan sloj.

        Svaka funkcija čuvaće memoriju za svoj ulaz i parcijalne izvode greške po ulazu u funkciju. Funkcija za propagaciju
        signala unapred dobijaće kao argument "mesto" na koje treba upiše rezultat, tj. ndimenzionalni niz.

        NAPOMENA: u starijoj verziji koda ova klasa se nazivala AbstractLayer. U trenutnoj implementaciji ne postoji klasa
        apstraktni sloj i ukoliko se negde u komentarima i dalje pominje apstraktni sloj, možete podrazumevati da se misli
        na klasu Function.
    """

    def __init__(self, name: str = 'unnamed'):
        self._training = True
        self._inputs: xp.ndarray = None  # potrebno je da pamtimo poslednji ulaz u sloj da bismo mogli da računamo
        # parcijalne izvode kod prolaska unazad
        self.name = name  # zarad praćenja toka i lovljenja bagova korisno je da svaki sloj ima ime

    @abstractmethod
    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        pass

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val

    @property
    def parameters(self) -> tuple:
        return tuple()

    @parameters.setter
    def parameters(self, params: tuple):
        pass

    def forward(self, inputs: xp.ndarray) -> xp.ndarray:
        """Zadatak metode je da propagira signal unapred, tj. da na osnovu ulaznih podataka datih sa
            new_input generiše i vrati izlaze.
        """
        self._pre_fw(inputs)

        return self(inputs)

    def _pre_fw(self, inputs: xp.ndarray):
        if self._training:
            self._inputs = inputs

    @abstractmethod
    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        """ dEdO predstavlja ndimenzionalni niz sa parcijalnim izvodima greške po izlazu iz ovog sloja.
        Te parcijalne izvode ovaj sloj nije u stanju da izračuna. Njih računa naredni sloj i prosleđuje ih ovom sloju.
        Zadatak ovog sloja je da izračuna parcijalne izvode greške po ulazima u ovaj sloj - setimo se da su ulazi u ovaj
        sloj zapravo izlazi iz prethodnog sloja!
        U ovoj metodi treba takođe implementirati logiku koja izračunava i ostale parcijalne izvode koji su od značaja
        za ovaj sloj (recimo, greške po težinama ovog sloja)

        """
        pass
