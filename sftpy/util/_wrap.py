import numpy as np

from ..sim import Actor, Component



# TODO replace wrap with just making sure each Component wraps ?

def wrap_phi(phi: np.ndarray,
         nflux: int):
    phi[:nflux] = np.fmod(phi[:nflux] + 2*np.pi, 2*np.pi)


class WrapPhi(Actor, Component):
    """
    Class that wraps phi between 0 and 2pi. Imposes periodic boundary condition.
    """

    def simulate(self):
        wrap_phi(self._data.phi, self._data.nflux)


def wrap_theta(phi: np.ndarray,
         theta: np.ndarray,
         nflux: int):
    poles = np.nonzero(np.fabs(theta[:nflux] - np.pi / 2) > np.pi / 2)[0]
    theta[poles] = np.fabs(np.fabs(theta[poles] - np.pi) - np.pi)
    phi[poles] = np.fmod(phi[poles] + np.pi, 2*np.pi)


class WrapTheta(Actor, Component):

    """
    Class that wraps theta between 0 and theta. Imposes polar boundary condition.
    """

    def simulate(self):
        wrap_theta(self._data.phi, self._data.theta, self._data.nflux)

