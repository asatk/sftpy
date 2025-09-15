import abc

from sftpy import simrc as rc

from ..util import Logger

loglvl = rc["component.loglvl"]

class Component(metaclass=abc.ABCMeta):

    prefix = "[component]"

    def __init__(self, loglvl: int=loglvl):
        self._log = Logger(loglvl, self.prefix)

    def log(self, loglvl: int, msg: str):
        self._log.log(loglvl, msg)

    def plot(self, loglvl: int, fname: str, *plot_args, **plot_kwargs):
        self._log.plot(loglvl, fname, *plot_args, **plot_kwargs)
