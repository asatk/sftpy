import numpy as np

from sftpy.component import Component


class CarringtonRotation(Component):
    """
    Class for component that rotates surface by Carrington rotation rate.
    """
    def __init__(self,
                 dt: float,
                 crot: float=-14.18,
                 loglvl: int=0):
        super().__init__(loglvl=loglvl)
        self._crot = crot
        self._dt = dt
        self._crot_step = crot * dt / 86400 * np.pi / 180


    prefix = "[carrington]"

    def move(self,
             phi: np.ndarray,
             nflux: int):
        phi[:nflux] += self._crot_step

    def __call__(self,
                 phi: np.ndarray,
                 nflux: int):
        self.move(phi, nflux)