import abc
import numpy as np
import numpy.typing as npt

from ..sim import Component, Data, Timeline

from .constants import MULTIPLIER



class Activity(Component, metaclass=abc.ABCMeta):
    """
    Generic Component describing the strength of stellar magnetic activity.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data)
        self._multiplier = multiplier

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @abc.abstractmethod
    def strength(self) -> float:
        """
        Strength of stellar magnetic activity normalized to Solar maximum.
        """
        ...

    @abc.abstractmethod
    def latitude(self) -> float:
        """
        Average latitude at which emergence occurs.
        """
        ...

    # leaving this here in anticipation of active longitude studies
    # @abc.abstractmethod
    # def longitude(self) -> float:
    #     ...



class ScheduledActivity(Activity, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def strength_schedule(self) -> npt.NDArray[np.float64]:
        ...

    @abc.abstractmethod
    def latitude_schedule(self) -> npt.NDArray[np.float64]:
        ...