import abc
import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..util import MapMaker

binflux = rc["physics.binflux"]
dt = rc["general.dt"]
thr = rc["rwalk.thr"]
diffusion = rc["physics.diffusion"]
loglvl = rc["component.loglvl"]
phibins = rc["synoptic.phibins"]
thetabins = rc["synoptic.thetabins"]
rad = rc["physics.rad"]



class RandomWalk(Component, metaclass=abc.ABCMeta):
    """
    `RandomWalk` subjects sources to small motions in random directions.

    The step size depends on the diffusion model which can include the impacts
    of a source's flux and the local flux concentration on its motion.
    """

    prefix = "[rwalk]"

    def __init__(self,
                 dt: float=dt,
                 diffusion: float=diffusion,
                 loglvl: int=loglvl):
        """
        Create a `RandomWalk` component, optionally specifying the timestep,
        flux threshold for difftest, and diffusion coefficient.

        Parameters
        ----------
        dt : float
            Timestep (s).
        diffusion : float
            Diffusion coefficient for flux dispersal (km^2/s).
        loglvl : int
            Maximum logging level.
        """
        super().__init__(loglvl)
        self._dt = dt
        self._diffusion = diffusion

    @abc.abstractmethod
    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):
        """
        Move sources randomly on the surface of the star.

        Parameters
        ----------
        phi : np.ndarray
            Longitudes of sources (rad).
        theta : np.ndarray
            Co-latitudes of sources (rad).
        flux : np.ndarray
            Fluxes of sources (10^18 Mx).
        nflux : int
            Number of sources

        Returns
        -------
        synoptic_dil : np.ndarray
            Dilated synoptic map.

        """

    def _move_finish(self,
                     phi: np.ndarray,
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
        rphi = rng.uniform(high=2*np.pi, size=nflux)
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

        phi[:nflux] = np.mod(np.arctan2(y, x) + 2*np.pi, 2*np.pi)
        theta[:nflux] = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))



class RW0(RandomWalk):
    """
    Schrijver diffusion mode 0 - step size is independent of flux and local
    flux density.
    """

    prefix = "[rwalk-0]"

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        # flux-independent stepping distance
        step_val = np.sqrt(4 * self._diffusion * self._dt) / rad
        step = np.full(nflux, step_val, dtype=np.float64)

        self._move_finish(phi, theta, step, nflux)



class RW1(RandomWalk):
    """
    Schrijver diffusion mode 1 - step size is dependent on local flux density.
    """

    def __init__(self,
                 synmap: MapMaker,
                 thr: float = thr,
                 dt: float = dt,
                 diffusion: float = diffusion,
                 loglvl: int = loglvl):
        super().__init__(dt, diffusion, loglvl)
        self._synmap = synmap
        # threshold in gauss for map of abs flux density
        # flux density contour for plage perimeter
        self._thr = thr

    prefix = "[rwalk-1]"

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        synoptic = self._synmap.make_nesting_map(phi, theta, flux, nflux, self._thr, binflux)

        step = np.ones(nflux, dtype=np.float64)

        # step size is adjusted by the ratio of diffusion coefficients from
        # Schrijver and Martin 90; not done in conjunction w/ flux density
        # dependence

        # flatten 2d abs flux map into 1d array.
        flat_map = np.ravel(np.fabs(synoptic))
        wherepx = np.where(flat_map > 0)

        ### determine the pixel in synoptic to which each source belongs

        # dimension 0 / phi
        phi_edges = np.histogram_bin_edges(
                phi[:nflux], bins=phibins, range=(0, 2*np.pi))
        phi_pixels = np.digitize(phi[:nflux], phi_edges) - 1
        
        # dimension 1 / theta
        theta_edges = np.histogram_bin_edges(
                theta[:nflux], bins=thetabins, range=(0, np.pi))
        theta_pixels = np.digitize(theta[:nflux], theta_edges) - 1

        # calculate 1d pixel for each source
        pixels = thetabins * phi_pixels + theta_pixels

        # determine which source pixels belong to those matching criterion
        ind = np.isin(pixels, wherepx)

        # adjust step size by ratio of diffusion (ref help?)
        step[ind] = 110.0 / 250.0

        # flux-independent stepping distance
        step = np.sqrt(4 * self._diffusion * step * self._dt) / rad

        self._move_finish(phi, theta, step, nflux)

        return synoptic


class RW2(RandomWalk):
    """
    Schrijver diffusion mode 2 - step size is dependent on flux in source.
    """

    prefix = "[rwalk-2]"
    
    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):

        # flux-independent stepping distance
        step_val = np.sqrt(4 * self._diffusion * self._dt) / rad

        # if flux-dependent steps are required, apply correction to concentrations
        # contained in a plage. See Schrijver+ 96 and PhD thesis Hagenaar p119 f7.4.
        step = step_val * 240. / 140. * np.exp(-np.abs(flux[:nflux])*binflux/35.)

        self._move_finish(phi, theta, step, nflux)
