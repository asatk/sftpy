from typing import Callable

import numpy as np
import numpy.typing as npt

from ..sim import Actor

from .constants import T_START



class Timeline(Actor):
    """
    Class for _abcs that tracks time through simulation.
    """

    def __init__(self,
                 step_sizes: list[tuple[int, float]],
                 t_start: float = T_START):
        self._t_start = t_start
        self._step = 0
        self._schedule_sim(step_sizes, t_start)
        self._schedule_library: list[npt.NDArray] = []
        self._callback_library: dict[Callable, int] = {}

    def _schedule_sim(self,
                      step_sizes: list[tuple[int, float]],
                      t_init: float) -> None:

        # determine schedule of timesteps
        dt_list = []
        nstep = 0
        for pair in step_sizes:

            # unpack step size data
            nstep_i = pair[0]
            dt_i = pair[1]

            # schedule each step's size according to the number prescribed
            dt_list_i = [dt_i] * nstep_i
            dt_list.extend(dt_list_i)

            # increase total number of steps to take
            nstep += nstep_i

        # time step for each simulation step
        schedule = np.asarray(dt_list, dtype=np.float64)

        # calculate time in each step
        timeline = np.cumsum(schedule) + t_init

        # schedule of step sizes (s)
        self._dt_schedule = schedule

        # timeline of simulation for each step since beginning (s)
        self._time_schedule = timeline

        # total number of steps to execute
        self._nstep = nstep

    def create_schedule(self, func) -> Callable:

        # assign next ID to new schedule
        sid = len(self._schedule_library)

        # calculate the schedule given the dt / step size schedule
        schedule = func()

        # add schedule to library
        self._schedule_library.append(schedule)

        # function to check a schedule
        def check_schedule():
            s = self._schedule_library[sid]
            return s[self._step]

        # look up schedules by the (hashable) function
        self._callback_library[check_schedule] = sid

        # return schedule checker to caller
        return check_schedule

    def get_schedule(self, func) -> npt.NDArray:
        sid = self._callback_library[func]
        schedule = self._schedule_library[sid]
        return schedule

    @property
    def t_init(self) -> float:
        return self._t_start

    @property
    def nstep(self) -> int:
        return self._nstep

    @property
    def step(self):
        return self._step

    def dt_schedule(self, unit: str = "s"):
        _dt = self._dt_schedule
        if unit == "s":
            return _dt
        elif unit == "d":
            return _dt / 86400
        elif unit == "y":
            return _dt / 86400 / 365.25
        return _dt

    def dt(self, unit: str = "s") -> float:
        _dt = self._dt_schedule[self._step]
        if unit == "s":
            return _dt
        elif unit == "d":
            return _dt / 86400
        elif unit == "y":
            return _dt / 86400 / 365.25
        return _dt

    def time_schedule(self, unit: str = "s"):
        _time = self._time_schedule
        if unit == "s":
            return _time
        elif unit == "d":
            return _time / 86400
        elif unit == "y":
            return _time / 86400 / 365.25
        return _time

    def time(self, unit: str = "s"):
        _time = self._time_schedule[self._step]
        if unit == "s":
            return _time
        elif unit == "d":
            return _time / 86400
        elif unit == "y":
            return _time / 86400 / 365.25
        return _time

    def simulate(self) -> None:
        # advance the simulation by one step
        self._step += 1

    def __str__(self):
        return str(self._time_schedule[self._step])

    def __mul__(self, x):
        return self._time_schedule[self._step] * x

    def __truediv__(self, x):
        return self._time_schedule[self._step] / x

    def __floordiv__(self, x):
        return self._time_schedule[self._step] // x

    def __format__(self, formatstr):
        return format(self._time_schedule[self._step], formatstr)
