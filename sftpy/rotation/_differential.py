"""
Differential flow patterns per Schrijver 2001 (a). See Eq. A3
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy

from ..activity import SchrijverCycle
from ..sim import Actor, Component, Data, Timeline
from ..sim.constants import STAR_RADIUS

from .constants import ROT_PER, ROT_A, ROT_B, ROT_C, MULTIPLIER
from .constants import ROT_FLUX_THRESHOLD



# class SinLatPolynomial(DifferentialRotation):
#     def __init__(self,
#                  time: Timeline,
#                  data: Data,
#                  order: int | list[int],
#                  coeff: npt.ArrayLike,
#                  multiplier: float = MULTIPLIER,
#                  loglvl: int = LOGLVL):
#         """
#         Initializes a DifferentialFlow _abcs.
#
#         Parameters
#         ----------
#         dt : float
#             Timestep (s)
#         multiplier : float
#             Relative strength of differential rotation
#         loglvl : int
#             Maximum logging level
#         """
#         super().__init__(time, data, multiplier, loglvl)
#
#         if isinstance(order, int):
#             order = np.arange(order + 1)
#         self._order = order
#
#         coeff = np.ravel(coeff)
#
#         if len(order) != len(coeff):
#             raise ValueError("Order and coefficients must have same length")
#
#         f = lambda lat: np.sum(np.power.outer(np.sin(lat), order) * coeff, axis=1)



class SinPolyDR(Actor, Component):
    """
    Schrijver differential mode 1 -
    """

    # MDI measurements have ROT_A increased by 0.2 (mode 2)

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 rot_a: float = ROT_A,
                 rot_b: float = ROT_B,
                 rot_c: float = ROT_C,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data)
        self._multiplier = multiplier

        # deg / day -> rad / step
        def schedule_step_size():
            return multiplier * time.dt_schedule("d") * np.pi / 180

        self._get_step_size = time.create_schedule(schedule_step_size)
        self._rot_a = rot_a
        self._rot_b = rot_b
        self._rot_c = rot_c

    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        nflux = data.nflux

        step_size = self._get_step_size()

        sinlat2 = np.cos(theta[:nflux])**2
        profile = self._rot_a + self._rot_b * sinlat2 + self._rot_c * sinlat2 ** 2
        phi[:nflux] += profile * step_size



class SinPolyTaperDR(SinPolyDR):
    """
    Schrijver differential mode 3 -
    """

    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        nflux = data.nflux

        step_size = self._get_step_size()

        sinlat2 = np.cos(theta[:nflux])**2
        # taper high-latitude differential rotation for simulation of of AB Dor-like star
        factor = (0.8 * np.exp(-np.abs(np.pi/2 - theta[:nflux]) / (80/180*np.pi)) + 0.2)
        profile = self._rot_a + (self._rot_b * sinlat2 + self._rot_c * sinlat2**2) * factor
        phi[:nflux] += profile * step_size



# TODO finish and test this class
class SinPolyExpandingFluxDR(Actor, Component):
    """
    Schrijver differential mode 4 -
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 cycle: SchrijverCycle,
                 rot_per: float = ROT_PER,
                 thr: float = ROT_FLUX_THRESHOLD,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data)
        self._multiplier = multiplier
        self._cycle = cycle
        self._rot_per = rot_per
        self._thr = thr


    def simulate(self):

        thr = self._thr

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux
        
        # TODO make this work
        dsource = self._cycle.strength() / self._cycle.multiplier

        cyclepol = np.sign(dsource)
        srcdiff = self._multiplier

        mdiff = 1.0     # magnitude of differential rotation
        diff = np.abs(self._multiplier)    # magnitude of the relative flow
        sdiff = np.sign(srcdiff)
        
        # deg / day -> rad / step
        scale = self._time.dt("d") * np.pi / 180

        a = self._rot_per
        b = -2.00
        c = -2.09

        sinlat2 = np.cos(theta[:nflux])**2
        profile = a + b * sinlat2 + c * sinlat2**2
        phi[:nflux] += profile * scale * mdiff

        # fast flux and slow flux?
        if sdiff > 0:
            fflux = (flux >  thr) & (theta >  np.pi/6) & (theta < np.pi/2) | \
                    (flux < -thr) & (theta >= np.pi/2) & (theta < np.pi*5/6)
            sflux = (flux >  thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                    (flux < -thr) & (theta >  np.pi/6) & (theta < np.pi/2)
        else:
            fflux = (flux < -thr) & (theta >  np.pi/6) & (theta < np.pi/2) | \
                    (flux >  thr) & (theta >= np.pi/2) & (theta < np.pi*5/6)
            sflux = (flux < -thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                    (flux >  thr) & (theta >  np.pi/6) & (theta < np.pi/2)

        scale2 = 10.0 / (STAR_RADIUS * 1000) * self._time.dt("d")

        phi[fflux] += scale2 * diff
        phi[sflux] -= scale2 * diff


        
        # TODO -- move map to main if showmap
        synmap(phi, theta, flux, fflux, sflux, nflux)

    # TODO move synmap to main.py

def _synmap(phi: np.ndarray,
            theta: np.ndarray,
            flux: np.ndarray,
            fflux: np.ndarray,
            sflux: np.ndarray,
            nflux: int,
            phibins: int=360,
            thetabins: int=180):

    # see fits2map for converting full-disk magnetogram FITS file into map and plotting
    # https://hesperia.gsfc.nasa.gov/rhessidatacenter/complementary_data/maps/

    ssflux = np.copy(flux)
    ssflux[np.abs(ssflux) < 50] = 0

    # TODO synoptic map prescription in schrijver + np.hist2d
    hist, _, _ = np.histogram2d(phi, theta, bins=(phibins, thetabins),
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
