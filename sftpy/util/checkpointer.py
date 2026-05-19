from typing import Dict, Any
import numpy as np

from .timestep import Timestep



def save_config(config: Dict[str, Any]):
    if config is not None:
        pass



class Checkpointer:

    def __init__(self,
                 filename: str,
                 frequency: int,
                 timestep: Timestep,
                 *callbacks):
        self._filename = filename
        self._frequency = frequency
        self._timestep = timestep
        self._callbacks = callbacks

    def checkpoint(self):
        if self._timestep.getstep() % self._frequency == 0:
            for c in self._callbacks:
                c()

class MapSaver:

    def __init__(self,
                 frequency: int,
                 timestep: Timestep,
                 phibins: int,
                 thetabins: int,
                 nsteps: int):
        self._frequency = frequency
        self._timestep = timestep
        self._phibins = phibins
        self._thetabins = thetabins
        self._nsteps = nsteps

        self._maps = np.empty(
            (nsteps // frequency + 1, phibins, thetabins),
            dtype=np.int64)

    @property
    def maps(self) -> np.ndarray:
        return self._maps

    def checkpoint(self,
             phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int):
        i = self._timestep.getstep()
        if i % self._frequency != 0:
            return

        sinlat = np.cos(theta[:nflux])
        map, _, _ = np.histogram2d(
            phi[:nflux], sinlat, weights=flux[:nflux],
            bins=(self._phibins, self._thetabins),
            range=((0, 2 * np.pi), (-1, 1)))
        self._maps[i // self._frequency] = map

    def save(self, filename: str):
        np.savez_compressed(filename, maps=self._maps)



class SpotSaver:

    def __init__(self,
                 frequency: int,
                 timestep: Timestep,
                 coord_type: np.dtype=np.float64,
                 count_type: np.dtype=np.int64):
        self._frequency = frequency
        self._timestep = timestep

        self._phi_record = np.empty(0, dtype=coord_type)
        self._theta_record = np.empty(0, dtype=coord_type)
        self._flux_record = np.empty(0, dtype=count_type)
        self._nflux_record = np.empty(0, dtype=count_type)

    def checkpoint(self,
                   phi: np.ndarray,
                   theta: np.ndarray,
                   flux: np.ndarray,
                   nflux: int):
        i = self._timestep.getstep()
        if i % self._frequency != 0:
            return

        self._phi_record = np.append(self._phi_record, phi)
        self._theta_record = np.append(self._phi_record, theta)
        self._flux_record = np.append(self._flux_record, flux)
        self._nflux_record = np.append(self._flux_record, nflux)

    def save(self, filename: str):
        np.savez_compressed(
            filename,
            phi_record=self._phi_record,
            theta_record=self._theta_record,
            flux_record=self._flux_record,
            nflux_record=self._nflux_record)


