import abc
import numpy as np

from ._actor import Actor
from ._component import Component
from ._data import Data
from ._timeline import Timeline



class Event(Actor, Component, metaclass=abc.ABCMeta):

    def __init__(self,
                 time: Timeline,
                 data: Data):
        super().__init__(time, data)

    @abc.abstractmethod
    def trigger(self) -> bool:
        ...

    @abc.abstractmethod
    def terminate(self) -> bool:
        ...



class ScheduledEvent(Event, metaclass=abc.ABCMeta):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 t_event: float):
        super().__init__(time, data)
        self._t_event = t_event

        # find simulation step immediately at or after event time
        t_diff = self._time.time_schedule() - self._t_event
        trigger_step = np.argwhere(t_diff >= 0)[0][0]
        self._trigger_step = trigger_step
        self._terminate_step = trigger_step

    @property
    def t_event(self):
        return self._t_event

    def trigger(self):
        if self._time.step == self._trigger_step:
            return True
        return False

    def terminate(self):
        if self._time.step >= self._terminate_step:
            return True
        return False



class RecurringEvent(ScheduledEvent, metaclass=abc.ABCMeta):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 t_first: float,
                 t_delta: float):
        super().__init__(time, data, t_first)
        self._t_first = t_first
        self._t_delta = t_delta

        time = self._time
        t_final = time.time_schedule()[-1]
        n_triggers = (t_final - self._t_first) / self._t_delta
        n_triggers = int(n_triggers) + 1

        if n_triggers == 1:
            print("INFO: This event will never recur.")

        trigger_steps = []
        for i in range(n_triggers):
            # find simulation step immediately at or after event time
            t_diff = time.time_schedule() - self._t_event - i * self._t_delta
            trigger_step = np.argwhere(t_diff >= 0)[0][0]
            trigger_steps.append(trigger_step)

        # only one trigger per timestep for a given event
        self._trigger_steps = np.unique(trigger_steps).tolist()

    @property
    def t_first(self):
        return self._t_first

    @property
    def t_delta(self):
        return self._t_delta

    def trigger(self):
        if self._time.step == self._trigger_steps[0]:
            self._trigger_steps.pop(0)
            return True
        return False

    def terminate(self):
        if len(self._trigger_steps) == 0:
            return True
        return False