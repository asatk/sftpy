import abc

from ._data import Data
from ._timeline import Timeline



class Component(metaclass=abc.ABCMeta):
    """
    Abstract base class for a Component of the SFT computation sequence.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data):
        """
        Initialize the _abcs with a maximum log level of `loglvl`.
        """
        self._time = time
        self._data = data

    @property
    def time(self):
        return self._time

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
