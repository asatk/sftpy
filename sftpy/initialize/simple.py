import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from .initialize import Initialize

nfluxmax = rc["general.nfluxmax"]
lng1 = rc["initialize.simple.lng1"]
lng2 = rc["initialize.simple.lng2"]
lat1 = rc["initialize.simple.lat1"]
lat2 = rc["initialize.simple.lat2"]
init_flux = rc["initialize.simple.init_flux"]
inv_pol = rc["cycle.inv_pol"]

class InitSimple(Initialize):
    """
    Initialize according to Schrijver: two spots at same phi opposite theta
    """

    def __init__(self,
                 nfluxmax: int=nfluxmax,
                 lng1: float=lng1,
                 lng2: float=lng2,
                 lat1: float=lat1,
                 lat2: float=lat2,
                 init_flux: int=init_flux,
                 inv_pol: int=inv_pol):
        super().__init__(nfluxmax)
        self._lng1 = lng1
        self._lng2 = lng2
        self._lat1 = lat1
        self._lat2 = lat2
        self._init_flux = init_flux
        self._inv_pol = inv_pol

    def init(self):
        
        # TODO lol remember when i hated this
        dr = np.pi / 180

        phi = np.zeros(self._nfluxmax, dtype=np.float64)
        phi[0] = self._lng1 * dr
        phi[1] = self._lng2 * dr
        phi[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

        theta = np.zeros(self._nfluxmax, dtype=np.float64)
        theta[0] = self._lat1 * dr
        theta[1] = self._lat2 * dr
        theta[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

        flux = np.zeros(self._nfluxmax, dtype=np.float64)
        flux[0] = self._init_flux * self._inv_pol
        flux[1] = -self._init_flux * self._inv_pol

        nflux = 2

        return phi, theta, flux, nflux
