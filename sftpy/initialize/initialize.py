import abc
import numpy as np

from sftpy import simrc as rc

from ..component import Component

loglvl = rc["component.loglvl"]
nfluxmax = rc["general.nfluxmax"]

class Initialize(Component, metaclass=abc.ABCMeta):

    def __init__(self,
                 nfluxmax: int=nfluxmax,
                 loglvl: int=loglvl):
        super().__init__(loglvl)
        self._nfluxmax = nfluxmax

    @abc.abstractmethod
    def init(self):
        ...
