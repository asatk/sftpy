import abc
import numpy as np

from ..component import Component

class MeridionalFlow(Component, metaclass=abc.ABCMeta):
    """
    Base class for meridional flow component of computation sequence
    """

    prefix = "[mflow]"

    def __init__(self,
                 dt: float,
                 mer_mult: float=1.0,
                 loglvl: int=0):
        super().__init__(loglvl)
        self._dt = dt
        self._mer_mult = mer_mult
    
    @abc.abstractmethod
    def move(self,
             theta: np.ndarray,
             nflux: int):
        ...

class MFNone(MeridionalFlow):

    prefix = "[mflow-none]"

    def move(self,
             theta: np.ndarray,
             nflux: int):
        return


class MF1(MeridionalFlow): 

    prefix = "[mflow-1]"

    def __init__(self,
                 dt: float,
                 mer_mult: float=1.0,
                 loglvl: int=0):
        super().__init__(dt, mer_mult, loglvl)
        self._a = 12.9e-3 * dt / 7.e5 * mer_mult
        self._b = 1.4e-3 * dt / 7.e5 * mer_mult

    def move(self,
             theta: np.ndarray,
             nflux: int):

        # from km / s to rad / timestep
        th = theta[:nflux]
        theta[:nflux] += -self._a * np.sin(2*th) + self._b * np.sin(4*th)


class MF2(MeridionalFlow):

    prefix = "[mflow-2]"

    def __init__(self,
                 dt: float,
                 mer_mult: float=1.0,
                 loglvl: int=0):
        super().__init__(dt)
        self._a = 12.7e-3 * dt / 7.e5 * mer_mult

    def move(self,
             theta: np.ndarray,
             nflux: int):

        # from km / s to rad / timestep
        th = theta[:nflux]
        lat = np.pi/2 - th
        
        # TODO check op precedence on IDL's <, >
        # TODO check if IDL's <, > convert to int
        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = self._a * np.sin(2*lat)
        theta[:nflux] += - f1 * f2 * f3


class MF3(MeridionalFlow):

    prefix = "[mflow-3]"

    def __init__(self,
                 dt: float,
                 mer_mult: float,
                 loglvl: int=0):
        super().__init__(dt, mer_mult, loglvl)
        # from km / s to rad / timestep
        self._a = 12.7e-3 * dt / 7.e5
        self._mer_mult = mer_mult

    def move(self,
             theta: np.ndarray,
             nflux: int):

        th = theta[:nflux]
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = self._a * np.sin(2*lat) * (1 + 0.06 * self._mer_mult)
        t1 = f1 * f2 * f3
        t2 = np.sin(lat * 0.9) * np.exp(-(lat * 3)**2) * self._a * 4 * self._mer_mult
        theta[:nflux] += - t1 + t2


class MF4(MeridionalFlow):

    prefix = "[mflow-4]"
    
    def __init__(self,
                 dt: float,
                 cycle,
                 loglvl: int=0):
        super().__init__(dt, mer_mult=0.0, loglvl=loglvl)
        self._dt = dt
        self._cycle = cycle

    def move(self,
             theta: np.ndarray,
             nflux: int):

        # TODO make this work
        # dsource, _ = self._cycle(time) / cyl_mult
        dsource = 1.0
        srcmer = 2.0 - max(np.fabs(dsource))

        # from km / s to rad / timestep
        th = theta[:nflux]

        # same as 2 but prints and smth in kit.pro (main.py)
        if abs(srcmer) < 1.e-5:
            return

        a = 12.7e-3 * self._dt / 7.e5 * srcmer
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat)
        theta[:nflux] += - f1 * f2 * f3

        self.log(1, f"******* {srcmer}")
