import abc

from sftpy import simrc as rc

from ..util import Logger

loglvl = rc["component.loglvl"]

class Component(metaclass=abc.ABCMeta):
    """
    Abstract base class for a component of the SFT computation sequence.
    Every component can log to stdout and plot using `matplotlib` functions
    and keywords.

    """

    prefix = "[component]"
    """
    Prefix to any log message for a given component.
    """

    def __init__(self, loglvl: int=loglvl):
        """
        Initialize the component with a maximum log level of `loglvl`.
        """
        self._log = Logger(loglvl, self.prefix)

    def log(self, loglvl: int, msg: str):
        """
        Output a log message to stdout.
        """
        self._log.log(loglvl, msg)

    def plot(self, loglvl: int, fname: str, *plot_args, **plot_kwargs):
        """
        Execute a `matplotlib.pyplot` function and keywords. The plot is not
        rendered until a call to `plt.show` is made; by default this is
        blocking. In order to prevent the blocking behavior of `plt.show`,
        pass the keyword block=False, displaying all figures immediately.
        """
        self._log.plot(loglvl, fname, *plot_args, **plot_kwargs)
