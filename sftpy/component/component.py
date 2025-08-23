import abc

from ..util import Logger

class Component(metaclass=abc.ABCMeta):

    prefix = "[component]"

    def __init__(self, loglvl: int=0):
        self._log = Logger(loglvl, self.prefix)

    def log(self, loglvl: int, msg: str):
        self._log.log(loglvl, msg)
