import numpy as np

from ..activity import Activity
from ..sim import Actor, Component, Data, Timeline
from ..sim.constants import STAR_RADIUS

from .constants import MULTIPLIER



class SinMF(Actor, Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data)

        self._multiplier = multiplier

        def _schedule_a():
            return 12.9e-3 * time.dt_schedule() / STAR_RADIUS * multiplier

        def _schedule_b():
            return 1.4e-3 * time.dt_schedule() / STAR_RADIUS * multiplier

        self._get_a = time.create_schedule(_schedule_a)
        self._get_b = time.create_schedule(_schedule_b)

    @property
    def multiplier(self):
        return self._multiplier

    def simulate(self):

        data = self._data
        theta = data.theta
        nflux = data.nflux

        a = self._get_a()
        b = self._get_b()

        # from km / s to rad / timestep
        th = theta[:nflux]
        theta[:nflux] = th - a * np.sin(2*th) + b * np.sin(4*th)



class SinTaperMF(Actor, Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data)

        self._multiplier = multiplier

        def _schedule_a():
            return 12.7e-3 * time.dt_schedule() / 7.e5 * multiplier

        self._get_a = time.create_schedule(_schedule_a)

    @property
    def multiplier(self):
        return self._multiplier

    def simulate(self):

        data = self._data
        theta = data.theta
        nflux = data.nflux

        a = self._get_a()

        # from km / s to rad / timestep
        th = theta[:nflux]
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat)
        theta[:nflux] = th - f1 * f2 * f3



class SinTaperEquatorModifiedMF(Actor, Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data)

        self._multiplier = multiplier

        def _schedule_a():
            return 12.7e-3 * time.dt_schedule() / 7.e5

        self._get_a = time.create_schedule(_schedule_a)

    @property
    def multiplier(self):
        return self._multiplier

    def simulate(self):

        data = self._data
        theta = data.theta
        nflux = data.nflux

        a = self._get_a()
        mult = self._multiplier

        th = theta[:nflux]
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat) * (1 + 0.06 * mult)
        t1 = f1 * f2 * f3
        t2 = np.sin(lat * 0.9) * np.exp(-(lat * 3)**2) * a * 4 * mult
        theta[:nflux] = th - t1 + t2



class SinTaperAmplifiedMF(Actor, Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 activity: Activity,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data, logging)

        self._multiplier = multiplier
        self._activity = activity

        # TODO check this
        def _schedule_a():
            dt = self._time.dt_schedule()
            strength = self._activity.strength()
            # normalize activity strength to 1
            # TODO max ???
            srcmer = 2.0 - np.max(np.abs(strength)) / self._activity.multiplier

            # from km / s to rad / timestep
            return 12.7e-3 * dt / STAR_RADIUS * srcmer

        self._get_a = time.create_schedule(_schedule_a)

    @property
    def multiplier(self):
        return self._multiplier

    def simulate(self):

        data = self._data
        theta = data.theta
        nflux = data.nflux

        a = self._get_a()

        th = theta[:nflux]
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat)
        theta[:nflux] = th - f1 * f2 * f3
