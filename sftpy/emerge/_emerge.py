import numpy as np

from ..regions import MagneticRegion
from ..sim import Actor, Component, Data, rng, Timeline

from .constants import EMERGE_USE_SHOT_NOISE
from ._nesting import PlageNests



class SchrijverEmerge(Actor, Component):
    """
    Emerges magnetic regions instantaneously. Has the capability to nest
    active regions within existing plages. Can also add Poisson noise to
    each concentration's emerged flux to mimic observations and closely
    replicate the flux distribution seen for the Sun.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 regions: list[MagneticRegion],
                 nest: PlageNests|None = None,
                 use_shot_noise: bool = EMERGE_USE_SHOT_NOISE):
        super().__init__(time, data)
        self._regions = regions
        self._nest = nest
        self._use_shot_noise = use_shot_noise
        self._data_new = Data()

    @property
    def regions(self) -> list[MagneticRegion]:
        return self._regions

    def simulate(self):

        data_new = self._data_new
        data_new.clear()

        # TODO multithread for computation of multiple cycles?
        # emerge new spots in each time step
        for region in self.regions:

            if np.abs(region.activity.strength()) < 1e-5:
                continue

            # locate emergence positions of new spots
            data_region = region.place_new_spots()

            # nest new spots within existing active regions
            if self._nest is not None:
                self._nest.place_active_regions(data_region)

            # fill in regions with flux point sources
            data_spots = region.fill_new_spots(data_region)

            # add Poisson statistical uncertainty in flux to mimic observations
            if self._use_shot_noise:
                nflux = data_spots.nflux
                flux = data_spots.flux[:nflux]
                scale_noise = np.sqrt(np.abs(flux))
                noise = rng.normal(size=nflux) * scale_noise
                noise = noise.astype(np.int64)
                flux += noise

            # add flux sources to data for all new spots emerged this step
            data_new.append_data(data_spots)

        # add all flux sources emerged this step to data for all sources
        self._data.append_data(data_new)



# TODO complete this class
class SchrijverAssimilate(Component, Actor):
    """
    Component for BMR emergence that assimilates solar magnetogram data.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data):
        super().__init__(time, data)

    def simulate(self):
        pass
