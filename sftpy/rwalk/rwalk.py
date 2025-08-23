import abc
import cv2 as cv
import numpy as np

from ..component import Component
from ..constants import binflux


class RandomWalk(Component, metaclass=abc.ABCMeta):

    prefix = "[rwalk]"

    def __init__(self,
                 dt: float,
                 rng: np.random.Generator,
                 thr: float=40.0,
                 diffusion: float=300.0,
                 loglvl: int=0):
        super().__init__(loglvl)
        self._dt = dt
        self._rng = rng
        # threshold in gauss for map of abs flux density
        # flux density contour for plage perimeter
        self._thr = thr
        # diffusion coefficient (cm/s^2 ??)
        self._diffusion = diffusion

    @abc.abstractmethod
    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             synoptic: np.ndarray,
             source: float=1.0, # TODO is this "source" the cycle "source" or other
             difftest: np.ndarray=None):
        ...

    def _move_start(self,
                    synoptic: np.ndarray,
                    source: float=1.0):

        # TODO is this "source" the cycle "source" or other
        if source < 0:
            # isn't np.fabs redundant?
            synoptic_abs = np.fabs(synoptic)
            temp = 10 / (binflux / 1.4752)
            # where is difftest used
            difftest = np.sum(synoptic_abs[synoptic_abs > temp]) * self._dt

        # threshold out flux to only include plages
        # binary map
        synoptic_thr = np.astype(synoptic > self._thr / (binflux / 1.4752), np.uint8)
        
        # TODO IDL -- compare smooth+dilation ops
        # smooth slightly and require at least 6 neighbors to be part of plage
        synoptic_sm = cv.filter2D(synoptic_thr, -1,
                                  np.ones((3,3), dtype=np.float64)/9)
        # dilate to add an extra ring of pixels to plage
        synoptic_thr2 = np.asarray(synoptic_sm > 5.9 / 9, dtype=np.uint8)
        synoptic_dil = cv.dilate(synoptic_thr2, np.ones((3,3), dtype=np.uint8))

        return synoptic_dil

    def _move_finish(self,
                     phi: np.ndarray,
                     theta: np.ndarray,
                     step: np.ndarray,
                     nflux: int):


        # move the sources given the above step size:
        # 1) rotate sphere so point is on the pole
        # 2) tilt over step in theta
        # 3) rotate over random phi angle
        # 4) apply inverse of 1
        # 5) step 1 approximated by letting concentrations move on a plane tanget
        # to the sphere at the pole, neglecting curvature

        # random direction to step in
        rphi = self._rng.uniform(high=2*np.pi, size=nflux)
        cosrphi = np.cos(rphi)
        sinrphi = np.sin(rphi)
        
        cosphi = np.cos(phi[:nflux])
        sinphi = np.sin(phi[:nflux])

        costheta = np.cos(theta[:nflux])
        sintheta = np.sin(theta[:nflux])

        crct = cosrphi * costheta
        x = cosphi * sintheta + step * (crct * cosphi - sinrphi * sinphi)
        y = sinphi * sintheta + step * (crct * sinphi + sinrphi * cosphi)
        z = costheta - step * cosrphi * sintheta

        phi[:nflux] = np.mod(np.arctan2(y, x) + 2*np.pi, 2*np.pi)
        theta[:nflux] = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))


class RWNone(RandomWalk):

    prefix = "[rwalk-none]"

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             synoptic: np.ndarray,
             source: float=1.0,
             difftest: np.ndarray=None):

        return synoptic


class RW0(RandomWalk):

    prefix = "[rwalk-0]"

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             synoptic: np.ndarray,
             source: float=1.0,
             difftest: np.ndarray=None):

        synoptic = self._move_start(synoptic, source)
        step = np.ones(nflux, dtype=np.float64)

        # evaluate the actual stepping distance
        step = np.sqrt(4 * self._diffusion * step * self._dt) / 7.e5

        self._move_finish(phi, theta, step, nflux)

        return synoptic


class RW1(RandomWalk):

    prefix = "[rwalk-1]"

    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             synoptic: np.ndarray,
             source: float=1.0,
             difftest: np.ndarray=None):

        synoptic = self._move_start(synoptic, source)
        step = np.ones(nflux, dtype=np.float64)

        # if required step size adjusted using ratio of diffusion coefficients
        # from Schrijver&Martin 90; not done in conjunction w/ flux-dens dep
        if difftest is not None:
            
            pix_x = np.clip(np.asarray(
                np.mod((phi[:nflux] + 2*np.pi), 2*np.pi) * 180 / np.pi,
                np.int64), a_min=0, a_max=359)
            pix_y = np.clip(np.asarray(
                (np.sin(theta[:nflux] + np.pi/2) + 1) * 90,
                np.int64), a_min=0, a_max=179)

            ind1 = synoptic[pix_x,pix_y] > 0

            step[ind1] = 110.0 / 250.0

        # evaluate the actual stepping distance
        step = np.sqrt(4 * self._diffusion * step * self._dt) / 7.e5

        self._move_finish(phi, theta, step, nflux)

        return synoptic


class RW2(RandomWalk):

    prefix = "[rwalk-2]"
    
    def move(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             synoptic: np.ndarray,
             source: float=1.0,
             difftest: np.ndarray=None):

        synoptic = self._move_start(synoptic, source)
        step = np.ones(nflux, dtype=np.float64)

        # evaluate the actual stepping distance
        step = np.sqrt(4 * self._diffusion * step * self._dt) / 7.e5

        # if flux-dependent steps are required, apply correction to concentrations
        # contained in a plage. See Schrijver+ 96 and PhD thesis Hagenaar p119 f7.4.
        step *= (240. / 140.) * np.exp(-np.fabs(flux[:nflux]) * binflux/35.)

        self._move_finish(phi, theta, step, nflux)

        return synoptic
