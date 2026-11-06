import cv2 as cv
import numpy as np
import numpy.typing as npt

from ..imgproc.constants import NPHI, NTHETA
from ..sim import Actor, Component, Data, rng, Timeline
from ..sim.constants import BINFLUX, STAR_RADIUS

from .constants import DIFFUSION, FLUX_THRESHOLD, STEP_DIST_PLAGE



def _move_finish(phi: np.ndarray,
                 theta: np.ndarray,
                 step: np.ndarray,
                 nflux: int):
    """
    Helper method that determines a random direction for each source to
    move.

    Parameters
    ----------
    phi : np.ndarray
        Longitudes of sources (rad).
    theta : np.ndarray
        Co-latitudes of sources (rad).
    step : np.ndarray
        Step size for each source.
    nflux : int
        Number of sources.
    """

    # move the sources given the above step size:
    # 1) rotate sphere so point is on the pole
    # 2) tilt over step in theta
    # 3) rotate over random phi angle
    # 4) apply inverse of 1
    # 5) step 1 approximated by letting concentrations move on a plane tanget
    # to the sphere at the pole, neglecting curvature

    # random direction to step in
    rphi = rng.uniform(high=2 * np.pi, size=nflux)
    cosrphi = np.cos(rphi)
    sinrphi = np.sin(rphi)

    # coordinates of sources
    cosphi = np.cos(phi[:nflux])
    sinphi = np.sin(phi[:nflux])
    costheta = np.cos(theta[:nflux])
    sintheta = np.sin(theta[:nflux])

    # nudge sources in random direction
    crct = cosrphi * costheta
    x = cosphi * sintheta + step * (crct * cosphi - sinrphi * sinphi)
    y = sinphi * sintheta + step * (crct * sinphi + sinrphi * cosphi)
    z = costheta - step * cosrphi * sintheta

    phi[:nflux] = np.fmod(np.arctan2(y, x), 2 * np.pi)
    theta[:nflux] = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))



class ConstantRW(Actor, Component):
    """
    Schrijver diffusion mode 0 - step size is independent of flux and local
    flux density.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 diffusion: float = DIFFUSION):
        super().__init__(time, data)

        self._diffusion = diffusion

        def _schedule_step_dist():
            return np.sqrt(4 * self._diffusion * time.dt_schedule()) / STAR_RADIUS

        self._get_step_dist = time.create_schedule(_schedule_step_dist)


    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        nflux = data.nflux

        step_dist = self._get_step_dist()

        # flux-independent stepping distance
        step_dist_all = np.full(nflux, step_dist, dtype=np.float64)

        _move_finish(phi, theta, step_dist_all, nflux)



class LocalFluxDensityRW(Actor, Component):
    """
    Schrijver diffusion mode 1 - step size is dependent on local flux density.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 # synmap: MapMaker,
                 flux_threshold: float = FLUX_THRESHOLD,
                 step_dist_plage: float = STEP_DIST_PLAGE,
                 diffusion: float = DIFFUSION):
        super().__init__(time, data)

        self._diffusion = diffusion
        # self._synmap = synmap
        # threshold in gauss for map of abs flux density
        # flux density contour for plage perimeter
        self._thr = flux_threshold / BINFLUX * 1.4752
        self._step_dist_plage = step_dist_plage

        # TODO replace with an img processing object
        # kernel for smoothing flux histogram
        nkernel = 3
        self._k_smooth = np.ones((nkernel, nkernel), dtype=np.float64)

        def _schedule_step_dist(dt: npt.NDArray[np.float64]):
            return np.sqrt(4 * self._diffusion * dt) / STAR_RADIUS

        self._get_step_dist = time.create_schedule(_schedule_step_dist)

    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        # TODO compare approaches and use faster one with img proc object

        # 2D histogram of unsigned flux in a sine latitude vs. longitude grid
        aflux = np.abs(flux[:nflux])
        costheta = np.cos(theta[:nflux])
        map_aflux, xedges, yedges = np.histogram2d(
            phi[:nflux], costheta, weights=aflux,
            bins=(NPHI, NTHETA),
            range=((0, 2 * np.pi), (-1., 1.)))

        # phi_binned = np.astype(np.digitize(phi[:nflux], xedges), np.int16)
        # theta_binned = np.astype(np.digitize(costheta, yedges), np.int16)
        # coord_binned = np.stack([phi_binned, theta_binned], axis=1)

        # threshold flux to include locations of plages -- binary map
        map_thr = np.asarray(map_aflux > self._thr, np.float64)

        # smooth slightly and require at least 6 neighbors to be part of plage
        map_smooth = cv.filter2D(map_thr, -1, self._k_smooth)

        # identify which concentrations have a flux-dependent step size
        # map_smooth[coord_binned] > 0.0

        step_dist = self._get_step_dist()
        step_dist_all = np.full(nflux, step_dist, dtype=np.float64)

        # step size is adjusted by the ratio of diffusion coefficients from
        # Schrijver and Martin 90; not done in conjunction w/ flux density
        # dependence

        # flatten 2d abs flux map into 1d array.
        wherepx = np.where(np.ravel(map_smooth) > 0)

        ### determine the pixel in synoptic to which each source belongs

        # dimension 0 / phi
        phi_edges = np.histogram_bin_edges(
                phi[:nflux], bins=NPHI, range=(0, 2*np.pi))
        phi_pixels = np.digitize(phi[:nflux], phi_edges) - 1
        
        # dimension 1 / theta
        theta_edges = np.histogram_bin_edges(
                theta[:nflux], bins=NTHETA, range=(0, np.pi))
        theta_pixels = np.digitize(theta[:nflux], theta_edges) - 1

        # calculate 1d pixel for each source
        pixels = NTHETA * phi_pixels + theta_pixels

        # determine which source pixels belong to those matching criterion
        ind = np.isin(pixels, wherepx)

        # adjust step size by ratio of diffusion (ref help?)
        step_dist_all[ind] = self._step_dist_plage

        _move_finish(phi, theta, step_dist_all, nflux)



class FluxRW(Actor, Component):
    """
    Schrijver diffusion mode 2 - step size is dependent on flux in source.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 diffusion: float = DIFFUSION):
        super().__init__(time, data)

        self._diffusion = diffusion

        def _schedule_step_dist():
            return 240. / 140. * np.sqrt(4 * self._diffusion * time.dt_schedule()) / STAR_RADIUS

        self._get_step_dist = time.create_schedule(_schedule_step_dist)

    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        # flux-independent stepping distance
        step_dist = self._get_step_dist()

        # if flux-dependent steps are required, apply correction to concentrations
        # contained in a plage. See Schrijver+ 96 and PhD thesis Hagenaar p119 f7.4.
        step_dist_all = step_dist * np.exp(-np.abs(flux[:nflux]) * BINFLUX / 35.)

        _move_finish(phi, theta, step_dist_all, nflux)
