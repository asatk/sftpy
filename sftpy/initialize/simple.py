import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from .initialize import Initialize
from ..emerge import BMRSchrijver

nfluxmax = rc["general.nfluxmax"]
lng1 = rc["initialize.simple.lng1"]
lng2 = rc["initialize.simple.lng2"]
lat1 = rc["initialize.simple.lat1"]
lat2 = rc["initialize.simple.lat2"]
flux1 = rc["initialize.simple.flux1"]
flux2 = rc["initialize.simple.flux2"]
inv_pol = rc["cycle.inv_pol"]
loglvl = rc["general.loglvl"]

class InitTwo(Initialize):
    """
    Initialize the SFT with two point sources anywhere on the stellar surface,
    the default behavior implemented by Schrijver's SFT.
    """

    def __init__(self,
                 nfluxmax: int=nfluxmax,
                 lng1: float=lng1,
                 lng2: float=lng2,
                 lat1: float=lat1,
                 lat2: float=lat2,
                 flux1: int=flux1,
                 flux2: int=flux2,
                 inv_pol: int=inv_pol,
                 loglvl: int=loglvl):
        """
        Create `InitSimple` component, optionally specifying the longitudes,
        co-latitudes, fluxes, and polarity of the two spots.

        Parameters
        ----------
        nfluxmax : int, optional
            The maximum number of spots to include in the simulation.
        lng1 : float, optional
            Initial longitude of the first spot (deg).
        lng2 : float, optional
            Initial longitude of the second spot (deg).
        lat1 : float, optional
            Initial co-latitude of the first spot (deg).
        lat2 : float, optional
            Initial co-latitude of the second spot (deg).
        flux1 : int, optional
            Initial flux of the first spot (10^18 Mx).
        flux2 : int, optional
            Initial flux of the second spot (10^18 Mx).
        inv_pol : int, optional
            Initial polarity of the stellar cycle.
        """
        super().__init__(nfluxmax, loglvl)
        self._lng1 = lng1
        self._lng2 = lng2
        self._lat1 = lat1
        self._lat2 = lat2
        self._flux1 = flux1
        self._flux2 = flux2
        self._inv_pol = inv_pol

    def init(self):
        
        # degrees to radians conversion
        dr = np.pi / 180

        # longitudes of two initial spots
        phi = np.zeros(self._nfluxmax, dtype=np.float64)
        phi[0] = self._lng1 * dr
        phi[1] = self._lng2 * dr
        phi[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

        # co-latitudes of two initial spots
        theta = np.zeros(self._nfluxmax, dtype=np.float64)
        theta[0] = self._lat1 * dr
        theta[1] = self._lat2 * dr
        theta[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

        # fluxes of two initial spots
        flux = np.zeros(self._nfluxmax, dtype=np.float64)
        flux[0] = self._flux1 * self._inv_pol
        flux[1] = self._flux2 * self._inv_pol

        # two spots
        nflux = 2

        return phi, theta, flux, nflux


class InitOne(Initialize):
    """
    Initialize the SFT with one extended spot using the mean Joy tilt to
    determine the orientation.
    """

    def __init__(self,
                 nfluxmax: int=nfluxmax,
                 lng1: float=lng1,
                 lat1: float=lat1,
                 flux1: int=flux1,
                 inv_pol: int=inv_pol,
                 loglvl: int=loglvl):
        super().__init__(nfluxmax, loglvl)
        self._lng1 = lng1
        self._lat1 = lat1
        self._flux1 = flux1
        self._inv_pol = inv_pol

    def init(self):

        nflux = 1

    ...