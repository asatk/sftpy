import numpy as np

from ..sim import Data, rng, ScheduledEvent, Timeline

from .constants import T_CONVERGE



class ConvergePolarCaps(ScheduledEvent):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 t_event: float = T_CONVERGE):
        super().__init__(time, data, t_event)
        self._time = time
        self._data = data

    def simulate(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        # randomly select concentrations that remain
        ind = rng.choice([False, True], size=nflux)

        # sum remaining concentrations
        nnew = np.sum(ind)

        # remove roughly half of
        netflux = np.sum(flux[:nflux][ind])
        flux[:nnew] = flux[:nflux][ind]
        phi[:nnew] = phi[:nflux][ind]
        theta[:nnew] = theta[:nflux][ind]

        # subtract off surplus net flux from one of the remaining spots
        flux[0] -= netflux

        data.nflux = nnew
