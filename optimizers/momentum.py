from typing import Dict

from optimizers.abstract_optimizer import Optimizer
from backend.backend import xp, address


class Momentum(Optimizer):
    """
    Jedan od problema gradijentnog spusta, koji nije svojstven samo treniranju neuronskih
    mreža, je često oscilovanje u blizini lokalnog minimuma. Oscilovanje se može smanjiti
    korišćenjem sledećeg ažuriranja
    vt = βvt−1 + α∇J(wt ),
    wt+1 = wt − vt .
    Parametar β se obično postavlja na vrednost 0.9. U slučaju oscilovanja, suprotne vrednosti
    gradijenata se u nekoj meri poništavaju, dok se kretanje dodatno ubrzava po pravcima
    kod kojih nema promene znaka. Momentum na taj način uspeva i da se, sa fiksiranim
    korakom učenja α, brže pomeri parametre u slučaju da se parametri nalaze na platou.

    VARIJANTA NESTEROVA
    Zbog akumulacije vrednosti i ubrzavanja učenja do koga dolazi kada uzastopne vred-
    nosti gradijenta budu istog znaka, momentum se nekada poredi sa kuglom koja se kotrlja
    nizbrdo i kojoj je potrebno vreme da uspori ili promeni smer kretanja.
    Kod Nesterovog momentuma, vrednost gradijenta ne računa se u tački wt , već u
    tački u koju bismo došli ako bismo pratili momentum, tj. u tački wt − βvt−1 . Na taj način
    Nestorov momentum može, u odnosu na klasični momentum, brže reagovati u situacijama
    kada kotrljanje kugle treba zaustaviti.
    Ažuriranje koje koristi Nesterov momentum dato je sa
    vt = βvt−1 + α∇J(wt − βvt−1 ),
    wt+1 = wt − vt.

    Primetimo da je jedina razlika u odnosu na momentum ta što se gradijent računa u
    pomerenoj tački.
    U radu Timothy Dozat. "Incorporating Nesterov Momentum into Adam” data je aproksimacija Nestorovog momentuma
    koja prilikom ažuriranja parametara parametre dodatno pomera, da bi u narednom koraku mogli da evaluiramo
    gradijent u trenutnoj tački. U radu je predloženo sledeće ažuriranje:

    gt = ∇J(wt ),
    vt = βvt−1 + αgt ,
    wt+1 = wt − (βvt + αgt ).
    """

    def __init__(self, lr: float = 0.005, beta: float = 0.9, nesterov: bool = False):
        super().__init__(lr)
        self.beta = beta
        self.v: Dict[int, xp.ndarray] = {}
        self.nesterov = nesterov  # s obzirom na sličnosti, ista klasa može vršiti i ažuriranje
        # u skladu sa Nesterovom modifikacijom momentuma.

    def update_parameters(self, params: xp.ndarray, grad: xp.ndarray) -> xp.ndarray:
        if address(params) not in self.v:
            self.v[address(params)] = xp.zeros_like(grad)

        v = self.v[address(params)]
        v = self.beta * v + self.lr * grad
        self.v[address(params)] = v

        """BITNA NAPOMENA!!!
            U narednim linijama izvršićemo ažuriranje prosleđenih parametara. Ažuriranja su tipično oblika
            w = w + alpha * update_step, pa istu liniju možemo zapisati i kao:
            w += alpha * update_step.
            Iako ove dve linije deluju ekvivalentno, zapravo postoji bitna razlika između njih.
            Razmotrimo ih detaljnije. 
            w = w + alpha * update_step.
            Ovom linijom nalažemo da se izračuna zbir w + alpha * update_step koji će biti smešten u privremeni ndarray objekat, a zatim će taj
            privremeni objekat biti dodeljen promenljivoj w, pa ćemo od privremenog učiniti trajni objekat. Bitno je naglasiti da memorijska adresa sadržaja
            od w pre izvršenja linije i memorijska adresa privremenog sadržaja w + alpha * update_step nisu jednake! Nakon izvršenja linije
            w = w + alpha * update_step, w će interno biti smešteno na novoj memorijskoj lokaciji, tj. kreirali smo novi tenzor, a prethodna vrednost od w
            se više ne koristi.
            
            Izvršenjem linije w += alpha * update_step, naglašavamo da izraz sa desne strane treba dodati na izraz sa leve strane i smestiti rezultat u 
            memorijski prostor koji je već zauzeo niz sa leve strane. Interna adresa na kojoj se niz zaista u memoriji čuva pre i nakon izvršenja linije 
            ostaće nepromenjena.
            
            U našem pristupu, gde tenzore parametara "raspoznajemo" na osnovu interne adrese, linija w = w + alpha * update_step dovešće do problema, tj. 
            uvek ćemo raditi sa novim adresama. Memorija za tenzore koji se ne koriste biće oslobođena i pristup adresi takvog tenzora dovešće do izuzetka. 
            Treniranje neće biti moguće. 
            Dakle, veoma je bitno da parametri ostanu na istoj adresi!!!
            (Ukoliko se računica ne može svesti na tenzor #= izraz (gde je sa # označen proizvoljni binarni operator), moramo da koristomo argument out= ili
            da vrednost privremenog rezultata na kraju eksplicitno kopiramo u parameters ndarray.  
        """

        if self.nesterov:
            params -= (self.beta * v + self.lr * grad)
        else:
            params -= v

        return params
