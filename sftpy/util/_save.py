import numpy as np
import os

from ..imgproc.constants import NPHI, NTHETA
from ..sim import Data, RecurringEvent, Timeline
from ..sim.constants import T_START

from . import constants as c
from .constants import SAVE_FREQ



class SaveMap(RecurringEvent):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 output_dir: str,
                 output_fstr: str | None = None,
                 nphi: int = NPHI,
                 ntheta: int = NTHETA,
                 t_first: float = T_START,
                 t_delta: int = SAVE_FREQ):
        super().__init__(time, data, t_first, t_delta)
        self._nphi = nphi
        self._ntheta = ntheta
        if output_fstr is None:
            output_fstr = "/map_{:06d}"
        self._outfile_template = output_dir + output_fstr

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    def simulate(self):

        data = self.data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        sinlat = np.cos(theta[:nflux])
        hist, _, _ = np.histogram2d(
            phi[:nflux], sinlat, weights=flux[:nflux],
            bins=(self._nphi, self._ntheta),
            range=((0, 2 * np.pi), (-1, 1)))

        step = self._time.step
        outfile_name = self._outfile_template.format(step)
        np.save(outfile_name, hist)



class SaveData(RecurringEvent):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 output_dir: str,
                 output_fstr: str | None = None,
                 t_first: float = T_START,
                 t_delta: int = SAVE_FREQ):
        super().__init__(time, data, t_first, t_delta)
        if output_fstr is None:
            output_fstr = "/data_{:06d}"
        self._outfile_template = output_dir + output_fstr


    def simulate(self):

        data = self.data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        step = self._time.step
        outfile_name = self._outfile_template.format(step)
        np.savez_compressed(outfile_name,
                            phi=phi, theta=theta, flux=flux, nflux=nflux)