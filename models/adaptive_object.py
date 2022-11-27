import warnings
from abc import abstractmethod

from layers.function import Function
from optimizers.abstract_optimizer import Optimizer


class AdaptiveObject(Function):
    """Objekti klase adaptivni objekat su funkcije koje mogu da adaptiraju neke svoje parametre ili komponente.
       Metoda koju svaka od izvedenih klasa mora da definiše je update_parameters.

       Ažuriranje parametara vršimo pomoću optimizatora (klase izvedene iz AbstractOptimizer)
    """

    def __init__(self, name: str = 'unnamed', optimizer: Optimizer = None):
        "Optimizator možemo, ali ne moramo proslediti konstruktoru prilikom kreiranja adaptivnog objekta"
        super().__init__(name)
        self._optimizer = optimizer

    @abstractmethod
    def update_parameters(self):
        pass

    def set_optimizer(self, optimizer: Optimizer, force: bool = False):
        """Želimo da ostvarimo fleksibilni pristup gde će biti moguće da se različiti slojevi adaptiraju
           drugačijim optimizatorima. Sa druge strane, želimo da imamo i mogućnost da koristimo istovetni optimizator
           za sve adaptivne objekte. Da ne bismo u tom slučaju eksplicitno postavljali optimizator svakom adaptivnom objektu,
           imaćemo kasnije, u klasi Network, način da dodelimo optimizator svim adaptivnim slojevima. Ideja parametra force
           je da "na silu" dodelimo optimizator adaptivnom sloju čak i u slučaju da je optimizator već postavljen.
           Kombinacijom ovih ideja mi možemo da, recimo, za 5. sloj postavimo jedan optimizator eksplicitno, a kasnije
           za celu mrežu postavimo neki drugačiji optimizator. Ukoliko je force==False, naš optimizator za 5. sloj će
           i dalje ostati nepromenjen.

           *** Jednostavnija varijanta je da prosto ne pitamo i da celokupno telo funkcije bude self._optimizer = deepcopy(optimizer) ***
        """
        if force or self._optimizer is None:
            self._optimizer = optimizer
        else:
            warnings.warn('Optimizer already set!')
