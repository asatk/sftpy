import cv2 as cv
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

    def make_nesting_map(self, phi, theta, flux, nflux, thr, binflux):

        aflux = np.abs(flux[:nflux])
        map, phi_edges, theta_edges = np.histogram2d(
            phi[:nflux], theta[:nflux], weights=aflux,
            bins=(phibins, thetabins),
            range=((0, 2*np.pi), (0, np.pi)))

        # threshold flux to include locations of plages -- binary map
        synoptic_thr = np.astype(map > thr / (binflux / 1.4752),
                                 np.uint8)

        # TODO IDL -- compare smooth+dilation ops
        # smooth slightly and require at least 6 neighbors to be part of plage
        synoptic_sm = cv.filter2D(synoptic_thr, -1,
                                  np.ones((3, 3), dtype=np.float64) / 9)

        # dilate to add an extra ring of pixels to plage
        synoptic_thr2 = np.asarray(synoptic_sm > 5.9 / 9, dtype=np.uint8)
        synoptic_dil = cv.dilate(synoptic_thr2, np.ones((3, 3), dtype=np.uint8))

        return synoptic_dil



    def make_Carrington_map(self,
                            phi: np.ndarray,
                            theta: np.ndarray,
                            flux: np.ndarray,
                            nflux: int):
        map, phi_edges, theta_edges = np.histogram2d(
            phi[:nflux], theta[:nflux], weights=flux[:nflux],
            bins=(phibins, thetabins),
            range=((0, 2 * np.pi), (0, np.pi)))
        return map

