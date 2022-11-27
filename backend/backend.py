try:
    import cupy as xp
except:
    import numpy as xp


def address(array) -> int:
    """Da bismo omogućili da jedan optimizator može da optimizuje više tenzora (ndimenztionalnih nizova parametara), moraćemo da
        izvršimo mapiranje iz tenzora u odgovarajuću istoriju koju optimizator treba da pamti (tipično momentum i sl).
        Nažalost, ne možemo koristiti dictionary kod koga je tip ključa ndarray (bilo da je u pitanju cp ili np), jer nije data implementacija
        za heširanje ndimenzionalnih nizova. Ono što možemo učiniti da ostvarimo mapiranje je da pristupimo samom pokazivaču koji se interno u
        ndimenzionalnom nizu čuva. Setite se, i numpy i cupy su biblioteke koje pružaju interfejs za korišćenje nizova i operacije nad njima, ali
        same implementacije izvršene su u C/C++ i CUDA C++. Internim podacima koji se tiču niza možemo da pristupimo pomoću __array_interface__, odnosno
        __cuda_array_interface__ atributa koji predstavlja dictionary koji mapira nazive relevantnih atributa u njihove vrednosti. """
    import numpy
    if isinstance(array, numpy.ndarray):
        return array.__array_interface__['data'][0]  # za numpy
    else:
        return array.__cuda_array_interface__['data'][0]  # za cupy
