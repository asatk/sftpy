import abc

from ..sim import Data, Timeline, ScheduledEvent
from ..sim.constants import T_START



class Initialize(ScheduledEvent, metaclass=abc.ABCMeta):

    def __init__(self,
                 time: Timeline,
                 data: Data):
        """
        Initialize the simulation with some arrangement of flux point sources.

        Parameters
        ----------
        nfluxmax : int, optional
            Maximum number of spots.
        """
        super().__init__(time, data, T_START)