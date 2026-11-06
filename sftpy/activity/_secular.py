import numpy as np

from ..sim import Timeline, Data

from ._activity import ScheduledActivity
from .constants import SEC_LAT, MULTIPLIER



class ActivityConstant(ScheduledActivity):
    """
    Schrijver activity mode 0: fixed activity level independent of time.
    """

    has_schedule = True

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 lat: float = SEC_LAT,
                 multiplier: float = MULTIPLIER):

        super().__init__(time, data, multiplier)

        self._strength = multiplier
        self._lat = lat

        self._strength_schedule = np.full(time.nstep, multiplier)
        self._latitude_schedule = np.full(time.nstep, lat)

    def strength(self):
        return self._strength

    def latitude(self) -> float:
        return self._lat

    def strength_schedule(self):
        return self._strength_schedule

    def latitude_schedule(self):
        return self._latitude_schedule
