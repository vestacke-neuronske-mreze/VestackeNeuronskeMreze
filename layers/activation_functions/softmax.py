from backend.backend import xp
from layers.activation_functions.activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """ f(x_i) =  exp(x_i) / sum_j exp(x_j), j iz {1, 2, 3, ... , n}
        Kod softmax funkcije i-ti ulaz utiče na svih n izlaza, pa je računanje
        parcijalnog izvoda za nijansu komplikovanije (treba razdvajati slučajeve)
        dFj/dXi = Si(1 - Sj) if i == j else -SiSj, Si = Sigmoid(xi)

        U praksi, softmax aktivaciona funkcija često je aktivaciona funkcija neurona u poslednjem sloju,
        a njen izlaz, koji čine brojevi iz (0, 1) koji u zbiru daju 1, tretiramo kao verovatnoće da uzorak pripada
        svakoj od n klasa (višekjlasna klasifikacija). Kriterijumska funkcija koju koristimo za višeklasnu klasifikaciju
        je kros entropija.

        Softmax aktivaciona funkcija nema adaptivne parametere, pa nam sam izvod dSoftmax/dX nije od značaja osim
        za računanje ostalih parcijalnih izvoda. Ono što nam je zaista potrebno je parcijalni izvod dE/dX.
        Za slučaj da je funkcija greške, E, kros entropija, a aktivaciona funkcjia poslednjeg sloja softmax,
        može se pokazati da se izvod dE/dX svodi na output - target, gde je target ciljna vrednost, a output
        izlaz neuronske mreže.

        *** S obzirom na sve što je gore navedeno, prostija implementacija mogla bi da "ignoriše" prolazak unazad
        i računa na to da će kod korišćenja kros entropije kao kriterijumske funkcije aktivaciona funkcija poslednjeg sloja
        biti upravo softmax, a odgovarajući parcijalni izvod računa direktno kao output - target ***
    """

    def __init__(self, name: str = "Softmax"):
        super().__init__(name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        y = xp.exp(inputs)
        tmp = xp.sum(y, axis=-1, keepdims=True)
        return xp.divide(y, tmp)

    def deriv(self, x: xp.ndarray) -> xp.ndarray:
        y = self(x)
        dx = xp.zeros((y.shape[0], y.shape[1], y.shape[1]))
        for batch_index in range(y.shape[0]):
            dx[batch_index, :, :] = -xp.matmul(y[batch_index, :].reshape(-1, 1),
                                               y[batch_index, :].reshape(-1, 1).T)
            dx[batch_index, :, :] += xp.diagflat(y[batch_index, :])

        return dx


