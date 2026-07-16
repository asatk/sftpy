import numpy as np

from sftpy import rng

from ..component import Component

class EmergeGradual(Component):

    prefix = "[gradual]"

    def __init__(self,
                 dt: float,
                 binflux: float,
                 loglvl: int=0):
        super().__init__(loglvl)

        self._dt = dt
        self._binflux = binflux

        self._laphi = np.zeros(0)
        self._latheta = np.zeros(0)
        self._laflux = np.zeros(0)
        self._latime = np.zeros(0)

    def emerge_gradual(self,
                       aphi: np.ndarray,
                       atheta: np.ndarray,
                       aflux: np.ndarray,
                       ntotal: int):
        # TODO vectorize this loop for gradual
        # gradual introduction of active regions

        laphi = self._laphi
        latheta = self._latheta
        laflux = self._laflux
        latime = self._latime

        for j in range(ntotal):

            # TODO add gradual emergence param
            dur = np.asarray(np.sum(np.abs(aflux)) * self._binflux / 0.05 / self._dt,
                             dtype=np.int64) + 1

            if dur > 1 or np.any((latime == 0) & (laflux != 0)):

                self.log(2, f"gradual flux emergence")

                an = len(aflux) / 2
                ai = rng.integers(high=dur, size=an)
                laphi = np.r_[laphi, aphi]
                latheta = np.r_[latheta, atheta]
                laflux = np.r_[laflux, aflux]
                latime = np.r_[latime, ai, ai]

                ia = (latime == 0) & (laflux != 0)
                if np.any(ia):
                    aphi = laphi[ia]
                    atheta = latheta[ia]
                    aflux = laflux[ia]
                    nadd = np.sum(ia) / 2
                else:
                    aphi = 0.0
                    atheta = 0.0
                    aflux = 0
                    nadd = 0

                ia = (latime != 0) & (laflux != 0)
                if np.any(ia):
                    laphi = laphi[ia]
                    latheta = latheta[ia]
                    laflux = laflux[ia]
                    latime = latime[ia]
                else:
                    laphi = 0.0
                    latheta = 0.0
                    laflux = 0
                    latime = 0

        # check this
        return aphi, atheta, aflux