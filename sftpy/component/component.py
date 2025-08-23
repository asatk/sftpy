import abc

from ..util import Logger

class Component(metaclass=abc.ABCMeta):

    prefix = "[component]"

    def __init__(self, loglvl: int=0):
        self._log = Logger(loglvl, self.prefix)

    def log(self, loglvl: int, msg: str):
        self._log.log(loglvl, msg)

    def plot(self, loglvl: int, fname: str, *plot_args, **plot_kwargs):
        self._log.plot(loglvl, fname, *plot_args, **plot_kwargs)
