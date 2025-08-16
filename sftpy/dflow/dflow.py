import abc
from matplotlib import pyplot as plt
import numpy as np

from ..cycle import Cycle

class DifferentialFlow(metaclass=abc.ABCMeta):
    """
    Base class for differential flow component of computation sequence.
    """

    def __init__(self,
                 dt: float,
                 dif_mult: float=1.0,
                 cyclepol: int=1):
        self._dt = dt
        self._dif_mult = dif_mult
        self._cyclepol = cyclepol

    @abc.abstractmethod
    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             **kwargs):
        ...


class DFNone(DifferentialFlow):

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):
        return


class DF1(DifferentialFlow):

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        if np.fabs(self._dif_mult) < 1.e-5:
            return

        mdiff = self._dif_mult

        # deg / day -> rad / step
        scale = self._dt * np.pi / 180 / 86400
        scale_mdiff = scale * mdiff

        a = (14.255984 - 14.18) * scale_mdiff
        b = -2.00 * scale_mdiff
        c = -2.09 * scale_mdiff

        sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
        phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a


class DF2(DifferentialFlow):

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        if np.fabs(self._dif_mult) < 1.e-5:
            return

        mdiff = self._dif_mult
        # deg / day -> rad / step
        scale = self._dt * np.pi / 180 / 86400
        scale_mdiff = scale * mdiff

        a = (14.255984 - 14.18 + 0.2) * scale_mdiff
        
        b = -2.00 * scale_mdiff
        c = -2.09 * scale_mdiff

        sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
        phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a


class DF3(DifferentialFlow):

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        if np.fabs(self._dif_mult) < 1.e-5:
            return

        mdiff = self._dif_mult
        # deg / day -> rad / step
        scale = self._dt * np.pi / 180 / 86400
        scale_mdiff = scale * mdiff

        a = (14.255984 - 14.18 + 0.2) * scale_mdiff
        
        b = -2.00 * scale_mdiff
        c = -2.09 * scale_mdiff

        sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
        phi[:nflux] += (b * sinlat2 + c * sinlat2**2) * \
            (0.8 * np.exp(-np.fabs(theta[:nflux]*180/np.pi - 90) / 80) + 0.2) + a


class DF4(DifferentialFlow):

    def __init__(self,
                 dt: float,
                 cycle: Cycle,
                 dif_mult: float=1.0,
                 thr: float=0.0,
                 cyclepol: int=1):
        super().__init__(dt, dif_mult, cyclepol)
        self._cycle = cycle
        self._thr = thr


    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        thr = self._thr
        mdiff = self._dif_mult

        if np.fabs(self._dif_mult) < 1.e-5:
            return
        
        # TODO make this work
        #dsource = self._cycle(time)[0] / cyl_mult
        dsource = 1.0

        # TODO cyclepolarity = -1 or 1... so index below is same
        self._cyclepol = (dsource[self._cyclepol] > 0) * 2 - 1
        srcdiff = self._cyclepol * self._dif_mult

        mdiff = 1.0     # magnitude of differential rotation
        diff = np.fabs(self._dif_mult)    # magnitude of the relative flow
        sdiff = np.sign(srcdiff)
        
        # deg / day -> rad / step
        scale = self._dt * np.pi / 180 / 86400
        scale_mdiff = scale * mdiff

        a = (14.255984 - 14.18) * scale_mdiff

        b = -2.00 * scale_mdiff
        c = -2.09 * scale_mdiff

        sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
        phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a

        # what if flux was limited to nflux elemetns instead of whole array

        # TODO this is wrong; fluxes move opposite dirs
        # fast flux and slow flux
        if sdiff > 0:
            fflux = (flux >  thr) & (theta >  np.pi/6) & (theta <   np.pi/2) | \
                    (flux < -thr) & (theta >= np.pi/2) & (theta < np.pi*5/6)
            sflux = (flux >  thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                    (flux < -thr) & (theta >  np.pi/6) & (theta <   np.pi/2)
        else:
            fflux = (flux < -thr) & (theta > np.pi*5/6) & (theta <  np.pi/2) | \
                    (flux >  thr) & (theta >=  np.pi/2) & (theta < np.pi*5/6)
            sflux = (flux < -thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                    (flux >  thr) & (theta >  np.pi/6) & (theta < np.pi/2)

        scale2 = 10.0 / 7e8 * dt
        scale2_diff = scale2 * diff

        phi[fflux] += scale2_diff
        phi[sflux] -= scale2_diff
        
        if kwargs.get("showmap"):
            self._synmap(phi, theta, flux, fflux, sflux, nflux)

    # TODO move synmap to main.py

    def _synmap(phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                fflux: np.ndarray,
                sflux: np.ndarray,
                nflux: int,
                phibins: int=360,
                thetabins: int=180):

        ssflux = np.copy(flux)
        ssflux[np.fabs(ssflux) < 50] = 0

        # TODO synoptic map prescription in schrijver + np.hist2d
        hist, _, _ = np.hist2d(phi, theta, bins=(phibins, thetabins),
                         weights=ssflux)
        hist = np.clip(hist, a_min=-50, a_max=50)
        h_min = np.min(hist)
        h_max = np.max(hist)

        # bytscl replacement
        hist_scale = np.asarray(((hist - h_min) / (h_max - h_min)) * 255, np.int64)
        # congrid replacement - 3x nearest-neighbor resampling of image
        hist_zoom = scipy.ndimage.zoom(hist_scale, 3, order=0)

        xf = phi[fflux] * 3 * 180 / np.pi
        yf = 3 * (np.sin(np.pi / 2 - theta[fflux]) * 90 + 89)
        xs = phi[sflux] * 3 * 180 / np.pi
        ys = 3 * (np.sin(np.pi / 2 - theta[sflux]) * 90 + 89)

        plt.imshow(hist_zoom / 4 + 96)
        plt.scatter(xf, yf, marker="+", color="black")
        plt.scatter(xs, ys, marker="-", color="red")
        plt.show()
