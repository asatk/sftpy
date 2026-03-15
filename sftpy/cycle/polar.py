import numpy as np

from sftpy import rng
from sftpy.component import Component
from sftpy.util import Timestep


class ConvergePolarCaps(Component):

    prefix = "[polar-converge]"

    def __init__(self,
                 t_cycle: float,
                 time: Timestep,
                 loglvl: int=0):
        super().__init__(loglvl=loglvl)
        self._t_cycle = t_cycle
        self._time = time
        self._ncycles = 0

    def converge(self,
                 phi: np.ndarray,
                 theta: np.ndarray,
                 flux: np.ndarray,
                 nflux: int):

        t = self._time.gettime() / 86400 / 365 - self._time.t_init
        tpdt = t + self._time.dt / 86400 / 365
        half_pd = self._t_cycle / 2
        if not (((t % half_pd) > (half_pd / 2)) and
                ((tpdt % half_pd) < (half_pd / 2))):
            return nflux

        ind = rng.uniform(size=nflux) < 0.5
        nnew = np.sum(ind)
        netflux = np.sum(flux[:nflux][ind])
        flux[:nnew] = flux[:nflux][ind]
        phi[:nnew] = phi[:nflux][ind]
        theta[:nnew] = theta[:nflux][ind]
        # TODO huh? "ensures zero total flux?
        flux[nnew//2] -= netflux

        return nnew
