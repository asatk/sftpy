from datetime import datetime

class Timestep():
    """
    Class for component that tracks time through simulation.
    """

    def __init__(self, dt: float, t_init: float=0.0):
        self._dt = dt
        self._time = t_init
        self._step = 0

    def getstep(self):
        return self._step

    def gettime(self):
        return self._time

    def step(self):
        self._time += self._dt
        self._step += 1

    def __call__(self):
        self.step()

    def __str__(self):
        return str(self._time)

    def __mul__(self, x):
        return self._time * x

    def __truediv__(self, x):
        return self._time / x

    def __floordiv__(self, x):
        return self._time // x

    def __format__(self, formatstr):
        return format(self._time, formatstr)
