import numpy as np

from ..sim import Actor, Component, Data, Timeline

from .constants import ROT_CAR



class CarringtonRotation(Actor, Component):
    """
    Class that rotates surface by Carrington rotation rate.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 rot_car: float = ROT_CAR):
        super().__init__(time, data)
        self._rot_car = rot_car

        def schedule_step_size():
            # deg / day -> rad / step
            return rot_car * time.dt_schedule("d") * np.pi / 180

        self._get_step_size = time.create_schedule(schedule_step_size)

    def simulate(self):

        phi = self._data.phi
        nflux = self._data.nflux

        step_size = self._get_step_size()

        phi[:nflux] += step_size
