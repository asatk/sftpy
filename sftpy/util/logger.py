from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt

from .timestep import Timestep



# singleton empty/null return value for clocks
null_time = timedelta(0)

class Logger:
    """
    Class for logging outputs for each component. Usually internal to each
    component.
    """

    def __init__(self, level: int, prefix: str):
        self._level = level
        self._prefix = prefix
        self._clocks = []
        self._clock_starts = {}
        self._clock_stops = {}
        self._clock_deltas = {}

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level: int):
        self._level = level

    def log(self, level: int, msg: str):
        if level <= self._level:
            print(f"{self._prefix} -- {msg}")

    def plot(self, level: int, func_name: str, *plot_args, **plot_kwargs):
        if level <= self._level:
            func = getattr(plt, func_name)
            func(*plot_args, **plot_kwargs)

    def pshow(self, level: int):
        if level <= self._level:
            plt.show()

    def clock_reset(self, c: int | str):
        if c in self._clocks:
            self._clocks.remove(c)
            self._clock_starts.pop(c)
            self._clock_stops.pop(c)
            self._clock_deltas.pop(c)
    
    def clock_start(self, c: int | str, msg: str=None):
        now = datetime.now()
        self._clock_starts[c] = now
        self._clock_stops[c] = now

        if c not in self._clocks:
            self._clocks.append(c)
            self._clock_deltas[c] = timedelta(0)

        if msg is not None:
            print(f"[> {msg} {now} <]")

        return now

    def clock_stop(self, c: int | str, msg: str=None):
        if c in self._clocks:

            now = datetime.now()
            delta = now - self._clock_starts[c]
            self._clock_stops[c] = now
            self._clock_deltas[c] += delta

            if msg is not None:
                print(f"[> {msg} {delta.total_seconds():.03f} <]")

            return delta

        return null_time

    def clock_check(self, c: int | str, msg: str=None):
        if c in self._clocks:

            if self._clock_starts[c] < self._clock_stops[c]:
                tcheck = self._clock_stops[c]
            else:
                tcheck = datetime.now()

            delta = tcheck - self._clock_starts[c]

            if msg is not None:
                print(f"[> {msg} {delta.total_seconds():.03f} s <]")

            return delta

        return null_time

    def clock_delta(self, c: int | str, msg: str=None):
        if c in self._clocks:

            delta = self._clock_deltas[c]

            if msg is not None:
                print(f"[> {msg} {delta.total_seconds():.03f} s <]")

            return delta

        return null_time

class TimedLogger(Logger):

    def __init__(self,
                 timestep: Timestep,
                 frequency: int,
                 level: int,
                 prefix: str):
        super().__init__(level, prefix)
        self._timestep = timestep
        self._frequency = frequency

    def log(self, level: int, msg: str):
        if self._timestep.getstep() % self._frequency == 0:
            super().log(level, msg)

    def plot(self, level: int, func_name: str, *plot_args, **plot_kwargs):
        if self._timestep.getstep() % self._frequency == 0:
            super().plot(level, func_name, *plot_args, **plot_kwargs)

    def pshow(self, level: int):
        if self._timestep.getstep() % self._frequency == 0:
            super().pshow(level)

    def clock_reset(self, c: int | str):
        if self._timestep.getstep() % self._frequency == 0:
            super().clock_reset(c)

    def clock_start(self, c: int | str, msg: str = None):
        if self._timestep.getstep() % self._frequency == 0:
            return super().clock_start(c, msg)
        return null_time

    def clock_stop(self, c: int | str, msg: str = None):
        if self._timestep.getstep() % self._frequency == 0:
            return super().clock_stop(c, msg)
        return null_time

    def clock_check(self, c: int | str, msg: str = None):
        if self._timestep.getstep() % self._frequency == 0:
            return super().clock_check(c, msg)
        return null_time

    def clock_delta(self, c: int | str, msg: str = None):
        if self._timestep.getstep() % self._frequency == 0:
            return super().clock_delta(c, msg)
        return null_time
