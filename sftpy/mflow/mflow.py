import abc
import numpy as np

from sftpy import simrc as rc

from ..component import Component

dt = rc["general.dt"]
mer_mult = rc["mflow.mult"]
loglvl = rc["component.loglvl"]
cyl_mult = rc["cycle.mult"]
rad = rc["physics.rad"]



class MeridionalFlow(Component, metaclass=abc.ABCMeta):
    """
    Base class for meridional flow component of computation sequence
    """

    prefix = "[mflow]"

    def __init__(self,
                 dt: float=dt,
                 mer_mult: float=mer_mult,
                 loglvl: int=loglvl):
        super().__init__(loglvl)
        self._dt = dt
        self._mer_mult = mer_mult
    
    @abc.abstractmethod
    def move(self,
             theta: np.ndarray,
             nflux: int):
        ...



class MF1(MeridionalFlow): 

    prefix = "[mflow-1]"

    def __init__(self,
                 dt: float=dt,
                 mer_mult: float=mer_mult,
                 loglvl: int=loglvl):
        super().__init__(dt, mer_mult, loglvl)
        # km/s -> rad/timestep
        self._a = 12.9e-3 * dt / rad * mer_mult
        self._b = 1.4e-3 * dt / rad * mer_mult

    def move(self,
             theta: np.ndarray,
             nflux: int):

        # from km / s to rad / timestep
        th = theta[:nflux]
        theta[:nflux] = th - self._a * np.sin(2*th) + self._b * np.sin(4*th)



class MF2(MeridionalFlow):

    prefix = "[mflow-2]"

    def __init__(self,
                 dt: float=dt,
                 mer_mult: float=mer_mult,
                 loglvl: int=loglvl):
        super().__init__(dt, mer_mult, loglvl)
        # km/s -> rad/timestep
        self._a = 12.7e-3 * dt / 7.e5 * mer_mult

    def move(self,
             theta: np.ndarray,
             nflux: int):

        # from km / s to rad / timestep
        th = theta[:nflux]
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = self._a * np.sin(2*lat)
        theta[:nflux] = th - f1 * f2 * f3



class MF3(MeridionalFlow):

    prefix = "[mflow-3]"

    def __init__(self,
                 dt: float=dt,
                 mer_mult: float=mer_mult,
                 loglvl: int=loglvl):
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
        theta[:nflux] = th - t1 + t2



class MF4(MeridionalFlow):

    prefix = "[mflow-4]"
    
    def __init__(self,
                 cycle,
                 dt: float=dt,
                 loglvl: int=loglvl):
        super().__init__(dt, mer_mult=0.0, loglvl=loglvl)
        self._cycle = cycle

    def move(self,
             theta: np.ndarray,
             nflux: int):

        dsource, _ = self._cycle.cycle()
        # normalize cycle strength to 1
        srcmer = 2.0 - np.max(np.abs(dsource)) / cyl_mult

        # from km / s to rad / timestep
        a = 12.7e-3 * self._dt / rad * srcmer
        th = theta[:nflux]
        lat = np.pi/2 - th

        f1 = 1 - np.exp(-np.clip(3 * th**3, a_min=-40, a_max=40))
        f2 = 1 - np.exp(-np.clip(3 * (np.pi-th)**3, a_min=-40, a_max=40))
        f3 = a * np.sin(2*lat)
        theta[:nflux] = th - f1 * f2 * f3
