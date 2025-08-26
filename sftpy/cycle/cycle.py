import abc
import numpy as np

from ..component import Component
from ..util import Timestep


class Cycle(Component, metaclass=abc.ABCMeta):

    prefix = "[cycle]"
    
    def __init__(self,
                 timestep: Timestep,
                 latlo: float=0.0,
                 lathi: float=25.0,
                 period: float=21.9,
                 ovr: float=3.0,
                 peak: float=4.0,
                 mult: float=1.0,
                 loglvl: int=0):
        super().__init__(loglvl)
        self._timestep = timestep
        self._latlo = latlo   # minimum latitude (deg) for source emergence
        self._lathi = lathi   # maximum latitude (deg) for source emergence
        self._pd = period     # cycle time
        self._ovr = ovr       # overlap between consecutive cycles
        self._peak = peak     # time of peak in activity and total flux
        self._mult = mult     # relative strength of activity cycle

    @abc.abstractmethod
    def cycle(self):
        ...

class CYCNone(Cycle):

    prefix = "[cycle-none]"

    def cycle(self):
        source = np.array([0.0], dtype=np.float64)
        latsource = np.array([90.0], dtype=np.float64)
        return source, latsource


class CYC3(CYCNone):
    """
    Component for emerging no regions at all.
    """

    prefix = "[cycle-3]"
    ...


class CYC0(Cycle):
    """
    Component for fixed-strength cycle independent of time.
    """

    prefix = "[cycle-0]"

    def cycle(self):
        source = np.array([self._mult], dtype=np.float64)
        latsource = np.array([self._lathi], dtype=np.float64)
        return source, latsource


class CYC1(Cycle):
    """
    Component for fixed-amplitude cycle
    """

    prefix = "[cycle-1]"

    def cycle(self):

        time = self._timestep.time()

        a = np.full(2, 2 * np.pi / (self._pd + 2 * self._ovr), dtype=np.float64)
        a[0] *= np.mod(time + self._pd, self._pd)
        a[1] *= np.mod(time + self._pd / 2, self._pd)

        if self._ovr < 0.01:
            b = 5
            c = 5.81
        elif np.abs(self._ovr - 2) < 0.01:
            b = 8
            c = 8.39
        else:
            a[0] = 2 * np.pi * self._peak / (self._pd + 2 * self._ovr)
            b = (a[0] * np.cos(a[0]) + np.sin(a[0])) / \
                    (2 * np.sin(a[0]) * a[0]**2) * np.pi**2
            c = a[0] * np.exp(-(a[0] / np.pi)**2 * b) / np.pi / \
                    np.clip(np.sin(a[0]), a_min=0, a_max=None)

        # clipped sin
        # sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

        # flagged sin
        sin_flag = (np.sin(a) > 0) + 0

        source = np.array([1, -1]) * sin_flag * self._mult * a / np.pi * \
                np.exp(-(a / np.pi)**2 * b) * c
        latsource = (self._lathi - (self._lathi - self._latlo) * a / np.pi) * sin_flag
        self.log(1, f"latsource {latsource}")

        return source, latsource


class CYC2(Cycle):
    
    prefix = "[cycle-2]"

    def cycle(self):

        time = self._timestep.time()

        a = np.full(2, 2 * np.pi / (self._pd + 2 * self._ovr), dtype=np.float64)
        a[0] *= np.mod(time + self._pd, self._pd)
        a[1] *= np.mod(time + self._pd / 2, self._pd)

        if self._ovr < 0.01:
            b = 5
            c = 5.81
        elif np.abs(self._ovr - 2) < 0.01:
            b = 8
            c = 8.39
        else:
            a[0] = 2 * np.pi * self._peak / (self._pd + 2 * self._ovr)
            b = (a[0] * np.cos(a[0]) + np.sin(a[0])) / \
                    (2 * np.sin(a[0]) * a[0]**2) * np.pi**2
            c = a[0] * np.exp(-(a[0] / np.pi)**2 * b) / np.pi / \
                    np.clip(np.sin(a[0]), a_min=0, a_max=None)

        # clipped sin
        # sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

        # flagged sin
        sin_flag = (np.sin(a) > 0) + 0

        source = np.array([1, -1]) * sin_flag * self._mult * a / np.pi * \
                np.exp(-(a / np.pi)**2 * b) * c
        latsource = (self._lathi - (self._lathi - self._latlo) * a / np.pi) * sin_flag

        # TODO yearssn
        source *= yearssn(time, yssn, ssn)

        return source, latsource


class CYC4(Cycle):

    prefix = "[cycle-4]"

    minima = np.array([
        1635.1, 1646.0, 1657, 1668, 1679, 1690, 1700, 1713.5, 1724.0, 1733.5,
        1745.0, 1756.0, 1767.0, 1775.5, 1784.0, 1798.5, 1811.0, 1825.0, 1833.5,
        1844.0, 1856.5, 1867.0, 1879.0, 1890.0, 1901.4, 1913.4, 1923.8, 1934.4,
        1944.3, 1954.3, 1964.5, 1976.6, 1986.5, 1996.7, 2006.9, 2018.2, 2029.2,
        2040.1]) - 1646.001

    # matchsolarrecords
    def cycle(self):

        time = self._timestep.time()
        minima = CYC4.minima
    

        polarity = np.mod(np.arange(len(minima), dtype=np.float64), 2) * 2 - 1

        # force solar conditions
        self._pd = 21.9
        self._ovr = 3.0
        self._latlo = 0.0

        # determine index of first cycle
        yi = np.nonzero(time < minima)[0][0] - 2
        yd = 2 * np.array([minima[yi+1] - minima[yi], \
                           minima[yi+2] - minima[yi+1]])

        a = np.pi * np.array([(time - minima[yi]) / (yd[0] / 2 + self._ovr), \
                              (time - minima[yi+1]) / (yd[1] / 2 + self._ovr)])
        # clipped sin
        # sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

        # flagged sin
        sin_flag = (np.sin(a) > 0) + 0

        source = self._mult * sin_flag * a / np.pi * polarity[[yi,yi+1]] * \
                np.exp(-(a / np.pi) ** 2 * 8) * 8.39
        latsource = (self._lathi - (self._lathi - self._latlo) * a / np.pi) * sin_flag

        if polarity[yi] < 0:
            source = np.roll(source, 1)
            latsource = np.roll(latsource, 1)

        # TODO yearssn
        source *= yearssn(time, yssn, ssn)

        return source, latsource
