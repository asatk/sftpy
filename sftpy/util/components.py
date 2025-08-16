import numpy as np

class WrapPhi():
    """
    Class for component that wraps phi between 0 and 2pi. Imposes periodic boundary condition.
    """

    def wrap(self,
             phi: np.ndarray,
             nflux: int):
        phi[:nflux] = np.fmod(phi[:nflux] + 2*np.pi, 2*np.pi)

    def __call__(self,
                 phi: np.ndarray,
                 nflux: int):
        self.wrap(phi, nflux)


class WrapTheta():

    """
    Class for component that wraps theta between 0 and theta. Imposes polar boundary condition.
    """

    def wrap(self,
             phi: np.ndarray,
             theta: np.ndarray,
             nflux: int):
        poles = np.nonzero(np.fabs(theta[:nflux] - np.pi / 2) > np.pi / 2)[0]
        theta[poles] = np.fabs(np.fabs(theta[poles] - np.pi) - np.pi)
        phi[poles] = np.fmod(phi[poles] + np.pi, 2*np.pi)

    def __call__(self,
                 phi: np.ndarray,
                 theta: np.ndarray,
                 nflux: int):
        self.wrap(phi, theta, nflux)


class Timestep():
    """
    Class for component that tracks time through simulation.
    """

    def __init__(self, dt: float, t_init: float=0.0):
        self._dt = dt
        self._time = t_init

    def time(self):
        return self._time

    def step(self):
        self._time += self._dt

    def __call__(self):
        self.step()

    def __str__(self):
        return str(self._time)

    def __format__(self, formatstr):
        return format(self._time, formatstr)
