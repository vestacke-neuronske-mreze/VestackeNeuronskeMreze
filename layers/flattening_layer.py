from backend.backend import xp
from layers.function import Function


class FlatteningLayer(Function):
    """U konvolucionoj mreži nakon konvolucionih, aktivacionih i možda pooling slojeva, dolaze
        "standardni" slojevi koje smo ranije imlpementirali - gusto povezani, aktivacioni i kriterijumske funkcije.
        Izlaz iz konvolucionog sloja (ili pooling sloja) je 4D tenzor, a ostatak merže radi sa matricama, pa
        je potrebno da nakon svih konvolucionih/pooling slojeva dodamo jedan sloj koji će taj 4D izlaz da pretvori u 2D matricu.
        To je zadatak ovog sloja.
        """
    def __init__(self, name: str = "Flattening Layer"):
        super().__init__(name)

    def __call__(self, inputs: xp.ndarray) -> xp.ndarray:
        return inputs.reshape((inputs.shape[0], -1))

    def backward(self, dEdO: xp.ndarray) -> xp.ndarray:
        # cilj prolaska unazad je da parcijalne izvode od matrice prevedemo u 4D tenzor.
        return dEdO.reshape(self._inputs.shape)
