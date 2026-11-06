import numpy as np

from ..sim import rng, Timeline, Data

from .constants import INIT_LONG1, INIT_COLAT1, INIT_FLUX1
from .constants import INIT_LONG2, INIT_COLAT2, INIT_FLUX2
from ._initialize import Initialize



class InitTwo(Initialize):
    """
    Initialize the SFT with two point sources anywhere on the stellar surface,
    the default behavior implemented by Schrijver's SFT.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 long1: float = INIT_LONG1,
                 long2: float = INIT_LONG2,
                 colat1: float = INIT_COLAT1,
                 colat2: float = INIT_COLAT2,
                 flux1: int = INIT_FLUX1,
                 flux2: int = INIT_FLUX2):
        """
        Initialize the simulation with two spots at the same longitude and
        opposite latitudes and fluxes.

        Parameters
        ----------
        long1 : float, optional
            Initial longitude of the first spot (deg).
        long2 : float, optional
            Initial longitude of the second spot (deg).
        colat1 : float, optional
            Initial co-latitude of the first spot (deg).
        colat2 : float, optional
            Initial co-latitude of the second spot (deg).
        flux1 : int, optional
            Initial flux of the first spot (10^18 Mx).
        flux2 : int, optional
            Initial flux of the second spot (10^18 Mx).
        """
        super().__init__(time, data)
        self._long1 = long1
        self._long2 = long2
        self._colat1 = colat1
        self._colat2 = colat2
        self._flux1 = flux1
        self._flux2 = flux2

    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux

        # degrees to radians conversion
        dr = np.pi / 180

        # longitudes of two initial spots
        phi[0] = self._long1 * dr
        phi[1] = self._long2 * dr
        phi[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

        # co-latitudes of two initial spots
        theta[0] = self._colat1 * dr
        theta[1] = self._colat2 * dr
        theta[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

        # fluxes of two initial spots
        flux[0] = self._flux1
        flux[1] = self._flux2

        # two spots
        data.nflux = 2