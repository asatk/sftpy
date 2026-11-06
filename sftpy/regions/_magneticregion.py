import abc

from ..activity import Activity
from ..sim import Component, Data, Timeline



class MagneticRegion(Component, metaclass=abc.ABCMeta):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 activity: Activity):
        super().__init__(time, data)
        self._activity = activity

    @property
    def activity(self) -> Activity:
        return self._activity

    @abc.abstractmethod
    def place_new_spots(self) -> Data:
        ...

    @abc.abstractmethod
    def fill_new_spots(self, data: Data) -> Data:
        ...
