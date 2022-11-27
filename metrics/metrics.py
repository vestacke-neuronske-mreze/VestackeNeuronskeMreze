from abc import ABC, abstractmethod

from backend.backend import xp
from layers.activation_functions.sigmoid import Sigmoid
from utils.utils import to_one_hot


class Metric(ABC):
    """Različite metrike korisne su kada evaluiramo rezultate treniranja ili pratimo napredak tokom treniranja.

        Najbitnija metoda je calculate_metric koja računa vrednost date metrike.
        Ideja je da se klasa koristi i za praćanje neke metrike tokom treniranja. S toga,
        potrebno je da budemo u stanju da izrađunamo vrednost neke metrike za celu epohu (calculate_metric zvaćemo
        za svaki batch, tj. više puta u istoj epohi). Tome služi metoda calculate_for_epoch.

        Na kraju, korisno je i da možemo da sagledamo kretanje vrednosti neke metrike na kraju treniranja. Možda možemo
        da vrednost metrike prikažemo na grafiku i sl. Zbog toga se vrednost metrike za svaku epohu čuvaju.
    """

    def __init__(self, _name: str):
        self.values_per_epoch = []
        self.name = _name

    @abstractmethod
    def calculate(self, y: xp.ndarray, t: xp.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_for_epoch(self) -> float:
        pass

    def last_epoch_value(self) -> float:
        return xp.round(self.values_per_epoch[-1], 4)  # radi lepšeg prikaza zaokružujemo rezultat na 4 decimale


class BinaryAccuracy(Metric):
    """Accuracy (~preciznost) predstavlja odnos između ispravno klasifikovanih uzoraka i ukupnog broja uzoraka.
       Kod klasifikacije preciznost je zapravo to što nas jedino i interesuje na kraju. Naravno, minimiziranjem
       odgovarajuće kriterijumske funkcije postižemo veću preciznost, ali za praćenje toka treniranja mnogo je
       korisnije da pratimo ovu metriku (accuracy od 0.982 je mnogo čitljivija i korisnija informacija nego
       recimo loss od 0.000342). """

    def __init__(self, from_logits: bool = False):
        super().__init__("accuracy")
        self.from_logits = from_logits
        self._count = 0
        self._sum = 0

    def calculate_for_epoch(self) -> float:
        a = self._sum / self._count
        self._sum = 0
        self._count = 0
        self.values_per_epoch.append(a)
        return a

    def calculate(self, y: xp.ndarray, t: xp.ndarray) -> float:
        """
            Ovde treba na osnovu network_output i target_value izračunati preciznost.
            Potrebno je međutim odvojeno tretirati preciznost kod binarne i višeklasne klasifikacije (ovde ne uzimamo
            u obzir višelablenu klasifikaciju, bar za sada). Razlog za to leži u tome kako su željene klase predstavljene
            u matrici target_value (a tome su prilagođeni i izlazi mreže, tj. network_output).

            Ukoliko se ne radi o višelabelnoj klasifikaciji svaki uzorak pripada tačno jednoj klasi, odnosno dodeljujemo
            svakom uzorku tačno jednu labelu.

            Kod binarne klasifikacije, imamo samo dve klase/labele i one su označene sa 0 i 1. Svakom uzorku dodeljujemo
            tačno jedan broj, 0 ili 1, u zavisnosti od klase kojoj pripada. Dakle, target_value je prosto matrica (~vektor)
            nula i jedinica dimenzija Nb x 1, gde je Nb broj primera u batch-u. Izlaz iz mreže tada je istih dimenzija,
            Nb x 1, i dobijen je tako što je na izlazu mreže primenjena sigmoidalna aktivaciona funkcija čiji je izlaz
            vrednost iz segmenta [0, 1]. Izlaz iz mreže predstavlja verovatnoću da dati uzorak pripada klasi 1.
            Dakle, ukoliko je izlaz veći od 0.5, računaćemo da uzorak treba svrstati u klasu 1. U suprotnom, svrstavamo
            ga u klasu 0 (znači da je verovatnoća da uzorak pripada klasi 0 veća ili jednaka od 0.5).
            Naravno, ukoliko bismo na izlazu dobili baš 0.5 ne bismo mogli da tvrdimo kojoj će klasi uzorak da pripada,
            ali s obzirom na (ne)preciznost izračunavanja, takoreći je nemoguće da na izlazu dobijemo baš 0.500000000..,
            tj. baš 0.5. Svejedno je kojoj ćemo klasi dodeliti uzorak ukoliko se ikada baš to desi.
            Da je uzorak ispravno klasifikovan znači da je abs(round(izlaz) - target_value) = 0
        """

        if self.from_logits:
            y = Sigmoid()(y)
        a = xp.sum(1 - xp.abs(xp.round(y) - t))

        self._sum += a
        self._count += y.size

        return float(a / y.size)


class Accuracy(Metric):
    """Accuracy (~preciznost) predstavlja odnos između ispravno klasifikovanih uzoraka i ukupnog broja uzoraka.
       Kod klasifikacije preciznost je zapravo to što nas jedino i interesuje na kraju. Naravno, minimiziranjem
       odgovarajuće kriterijumske funkcije postižemo veću preciznost, ali za praćenje toka treniranja mnogo je
       korisnije da pratimo ovu metriku (accuracy od 0.982 je mnogo čitljivija i korisnija informacija nego
       recimo loss od 0.000342).

        Kod višeklasne klasifikacije uobičajeno je da cijne vrednosti, tj. ciljnu klasu predstavimo drugačije.
        Ukoliko imamo k klasa i uzorak treba da pripadne, recimo klasi 5, to nećemo označiti na taj način da određena
        skalarna vrednost bude jednaka 4 (ili 5, zavisi brojimo li od nule).
        Tipično je da pripadnost klasi zadamo vektorom dužine k, a da na i-tom mestu u vektoru imamo verovatnoću
        da uzorak pripada baš i-toj klasi. Ukoliko uzorak pripada klasi 5, ciljna vrednost biće vektor dužine k
        takav je na 5. mestu u vektoru 1, a na svim ostalim mestima 0 - hoćemo da naša mreža nauči da bude sigurna
        da taj uzorak pripada baš klasi 5, sa verovatnoćom 1, i nikako bilo kojoj drugoj klasi.
        Dakle, ciljna vrednost, tj. vektor, bio bi [0, 0, 0, 0, 1, 0, ... , 0]. Ovakav način predstavljanja
        pripadnosti klasi naziva se one-hot enkodiranje (kao da je samo jedno mesto u vektoru "upaljeno", a ostala
        "ugašena").
        Na izlazu iz mreže, kod višeklasne klasifikacije, nalazi se softmax aktivaciona funkcija koja izlazne vektore
        svodi na vektore kod kojih je na svakom mestu vrednost iz [0, 1] uz uslov da je zbir vrednosti na svim mestima
        u vektoru jednak 1. Izlaz na i-tom mestu tretiramo kao verovatnoću da uzorak pripada i-toj klasi.
        Kod klasifikacije smatramo da uzorak pripada j-toj klasi ukoliko je verovatnoća da pripada j-toj klasi najveća
        (ponovo, u teoriji je moguće da na mestima i i j, i!=j, imamo podjednake verovatnoće, ali u praksi je takoreći
        nemoguće).
        Dakle, uzorak smatramo ispravno klasifikovanim ukoliko je argmax(izlaz) = argmax(target), tj. ako je
        argmax(izlaz) - argmax(target) = 0
        argmax je funkcija koja vraća indeks maksimuma u nizu (kod implementacije treba nravno imati na umu da
        radimo sa više primera odjednom, pa kod argmax treba navesti i dimenziju po kojoj tražimo maksimum).

    """

    def __init__(self, one_hot: bool = True):
        super().__init__("accuracy")
        self._count = 0
        self._sum = 0
        self.one_hot = one_hot

    def calculate_for_epoch(self) -> float:
        a = self._sum / self._count
        self._sum = 0
        self._count = 0
        self.values_per_epoch.append(a)
        return a

    def calculate(self, y: xp.ndarray, t: xp.ndarray) -> float:
        if not self.one_hot:
            t = to_one_hot(t, y.shape[-1])

        tmp1 = xp.argmax(y, axis=-1)
        tmp2 = xp.argmax(t, axis=-1)
        incorrect_classified = xp.count_nonzero(tmp1 - tmp2)
        num_of_samples = y.size / y.shape[-1]
        a = (num_of_samples - incorrect_classified)

        self._sum += a
        self._count += num_of_samples

        return float(a / num_of_samples)
