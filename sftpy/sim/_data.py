import numpy as np
import numpy.typing as npt

from .constants import NFLUXMAX

class Data:
    def __init__(self,
                 nfluxmax: int = NFLUXMAX):
        self._phi = np.zeros(nfluxmax, dtype=np.float64)
        self._theta = np.zeros(nfluxmax, dtype=np.float64)
        self._flux = np.zeros(nfluxmax, dtype=np.int64)
        self._nflux = 0

    @classmethod
    def make_data(cls,
                  phi: npt.NDArray[np.float64],
                  theta: npt.NDArray[np.float64],
                  flux: npt.NDArray[np.int64],
                  nflux: int,
                  nfluxmax: int = NFLUXMAX):
        data = Data(nfluxmax=nfluxmax)
        data._phi = phi
        data._theta = theta
        data._flux = flux
        data._nflux = nflux
        return data

    @property
    def phi(self) -> npt.NDArray[np.float64]:
        return self._phi

    @phi.setter
    def phi(self, val: npt.NDArray[np.float64]):
        self._phi = val

    @property
    def theta(self) -> npt.NDArray[np.float64]:
        return self._theta

    @theta.setter
    def theta(self, val: npt.NDArray[np.float64]):
        self._theta = val

    @property
    def flux(self) -> npt.NDArray[np.int64]:
        return self._flux

    @flux.setter
    def flux(self, val: npt.NDArray[np.int64]):
        self._flux = val

    @property
    def nflux(self) -> int:
        return self._nflux

    @nflux.setter
    def nflux(self, val: int):
        self._nflux = val

    def append(self, data: Data):
        nflux_add = data.nflux
        nflux_new = self._nflux + nflux_add

        if nflux_new >= NFLUXMAX:
            print("Too many spots emerged this timestep. Simulation aborting.")
            exit(1)

        self._phi[self._nflux:nflux_new] = data.phi[:nflux_add]
        self._theta[self._nflux:nflux_new] = data.theta[:nflux_add]
        self._flux[self._nflux:nflux_new] = data.flux[:nflux_add]
        self._nflux = nflux_new

    def append_sources(self,
                       phi: npt.NDArray[np.float64],
                       theta: npt.NDArray[np.float64],
                       flux: npt.NDArray[np.int64],
                       nflux: int):

        nflux_add = nflux
        nflux_new = self._nflux + nflux_add

        if nflux_new >= NFLUXMAX:
            print("Too many spots emerged this timestep. Simulation aborting.")
            exit(1)

        self._phi[self._nflux:nflux_new] = phi[:nflux_add]
        self._theta[self._nflux:nflux_new] = theta[:nflux_add]
        self._flux[self._nflux:nflux_new] = flux[:nflux_add]
        self._nflux = nflux_new

    def clear(self, n: int = 0):
        self._phi[n:self._nflux] = 0.0
        self._theta[n:self._nflux] = 0.0
        self._flux[n:self._nflux] = 0.0
        self._nflux = n