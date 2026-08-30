import numpy as np
from sftpy import simrc as rc

phibins = rc["viz.phibins"]
thetabins = rc["viz.thetabins"]

class MapMaker:

    def __init__(self,
                 phibins: int=phibins,
                 thetabins: int=thetabins):
        self._phibins = phibins
        self._thetabins = thetabins





    def make_Carrington_map(self,
                            phi: np.ndarray,
                            theta: np.ndarray,
                            flux: np.ndarray,
                            nflux: int):
        sinlat = np.cos(theta[:nflux])
        map, phi_edges, theta_edges = np.histogram2d(
            phi[:nflux], sinlat, weights=flux[:nflux],
            bins=(phibins, thetabins),
            range=((0, 2 * np.pi), (-1, 1)))
        return map

