class Timestep():
    """
    Class for component that tracks time through simulation.
    """

    def __init__(self, dt: float, t_init: float=0.0):
        self._dt = dt
        self._time = t_init

    def time(self):
        return self._time

    def step(self):
        self._time += self._dt

    def __call__(self):
        self.step()

    def __str__(self):
        return str(self._time)

    def __format__(self, formatstr):
        return format(self._time, formatstr)
