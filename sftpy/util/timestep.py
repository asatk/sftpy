from datetime import datetime

class Timestep():
    """
    Class for component that tracks time through simulation.
    """

    def __init__(self, dt: float, t_init: float=0.0, timer: int=0):
        self._dt = dt
        self._time = t_init
        self._step = 0
        self._timer = timer
        if timer:
            self._init_timer(timer)


    def _init_timer(self, timer: int):
        self._ticpu = datetime.now()
        print(f"[> Timer starts: {self._ticpu} <]")

    def step(self):
        return self._step

    def time(self):
        return self._time

    def readtimer(self, msg):
        tcpu = datetime.now()
        dtcpu = tcpu - self._ticpu
        print(f"[> {msg} {dtcpu.total_seconds():3f} s <]")

    def step(self):
        self._time += self._dt
        self._step += 1

        if self._timer and (self._step % self._timer == 0):
            self.readtimer(f"{self._step:d} steps in")

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
