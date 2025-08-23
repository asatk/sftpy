import numpy as np

from ..component import Component

class WrapPhi(Component):
    """
    Class for component that wraps phi between 0 and 2pi. Imposes periodic boundary condition.
    """

    prefix = "[wrap-phi]"

    def wrap(self,
             phi: np.ndarray,
             nflux: int):
        phi[:nflux] = np.fmod(phi[:nflux] + 2*np.pi, 2*np.pi)

    def __call__(self,
                 phi: np.ndarray,
                 nflux: int):
        self.wrap(phi, nflux)


class WrapTheta(Component):

    """
    Class for component that wraps theta between 0 and theta. Imposes polar boundary condition.
    """

    prefix = "[wrap-theta]"

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

