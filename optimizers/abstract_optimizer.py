from abc import ABC, abstractmethod

from backend.backend import xp


class Optimizer(ABC):
    """Optimizator ima za cilj da optimizuje parametre adaptivnog objekta (videti klasu AdaptiveObject).
       Metoda koju svaki konkretni, izvedeni, optimizator mora da implementira je update_parameters.
       Metoda kao argumente uzima parametre i odgovarajuće parcijalne izvode greške po tim parametrima.
       Na osnovu tih parcijalnih izvoda, optimizator ažurira parametre.

       O optimizaciji parametara neuronske mreže:

       Ukoliko posmatramo treniranje neuronske mreže kao optimizacioni problem,
       uočavamo par specifičnosti koje bi trebalo uzeti u obzir prilikom kreiranja efikasnog al-
       goritma učenja.

       - Gradijent se najčešće aproksimira na osnovu relativno malog broja primera iz sku-
            pa uzoraka za treniranje, pa odabir parametra α koristeći tehnike linijskog pre-
            traživanja (što je uobičajeno kod drugih problema numeričke optimizacije) ne bi
            doveo do zadovoljavajućeg rezultata (α parametar koji je "najbolji” za relativno
            mali broj primera ne mora biti dobar uzimajući u obzir ceo trening skup). Dakle,
            potrebno je na neki način osigurati da parametar α ne bude ni premali (jer bi
            to usporilo učenje), ni preveliki (što može dovesti do velike varijanse ažuriranja i
            takođe usporava učenje).

       - Kriterijumska funkcija često može imati veliki broj oblasti u kojima se njena vrednost
            veoma sporo menja. Kod takvih platoa vrednost gradijenta po svim pravcima
            je bliska nuli, ali algoritam ipak treba biti u stanju da "pobegne” iz takvih oblasti.

        - Tokom treniranja neuronske mreže određene sinapse se "specijalizuju” za odredđne
            odlike uzoraka iz skupa uzoraka za treniranje. Na primer, aktivacioni potencijal
            nekog neurona može biti blizak ili jednak nuli za većinu primera iz skupa uzoraka
            za treniranje, dok dostiže veće vrednosti za neke specifične primere koji ispoljavaju
            određenu odliku. Bilo bi dobro ukoliko bi algoritam bio u stanju da od "retkih”
            odlika uči brže.
       """

    def __init__(self, lr: float):
        self.lr = lr

    @abstractmethod
    def update_parameters(self, params: xp.ndarray,
                          grad: xp.ndarray) -> xp.ndarray:
        pass
