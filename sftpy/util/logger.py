from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt

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

    def log(self, level: int, msg: str):
        if level <= self._level:
            print(f"{self._prefix} -- {msg}")

    def plot(self, level: int, func_name: str, *plot_args, **plot_kwargs):
        if level <= self._level:
            func = getattr(plt, func_name)
            func(*plot_args, **plot_kwargs)

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
                print(f"[> {msg} {delta.total_seconds()} <]")

        return timedelta(0)

    def clock_check(self, c: int | str, msg: str=None):
        if c in self._clocks:

            if self._clock_starts[c] < self._clock_stops[c]:
                tcheck = self._clock_stops[c]
            else:
                tcheck = datetime.now()

            delta = tcheck - self._clock_starts[c]

            if msg is not None:
                print(f"[> {msg} {delta.total_seconds():3f} s <]")

            return delta

        return timedelta(0)

    def clock_delta(self, c: int | str, msg: str=None):
        if c in self._clocks:

            delta = self._clock_deltas[c]

            if self._clock_starts[c] >= self._clock_stops[c]:
                delta += datetime.now() - self._clock_starts[c]

            if msg is not None:
                print(f"[> {msg} {delta.total_seconds():3f} s <]")

        return timedelta(0)
