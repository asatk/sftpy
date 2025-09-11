from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt

class Logger():
    """
    Class for logging outputs for each component. Usually internal to each
    component.
    """

    def __init__(self, level: int, prefix: str):
        self._level = level
        self._prefix = prefix
        self._clocks = {}

    def log(self, level: int, msg: str):
        if level <= self._level:
            print(f"{self._prefix} -- {msg}")

    def plot(self, level: int, fname: str, *plot_args, **plot_kwargs):
        if level <= self._level:
            func = getattr(plt, fname)
            func(*plot_args, **plot_kwargs)
    
    def clockstart(self, c: int|str, msg: str=None):
        now = datetime.now()
        self._clocks[c] = now
        if msg is not None:
            print(f"[> {msg} {now} <]")
        return now

    def clockstop(self, c: int|str, msg: str=None):
        if c in self._clocks.keys():
            now = datetime.now()
            delta = now - self._clocks[c]
            if msg is not None:
                print(f"[> {msg} {delta.total_seconds():3f} s <]")
            return delta
        return timedelta(0)


