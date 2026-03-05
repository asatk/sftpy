"""
These components encapsulate the behavior of a stellar activity cycle as
determined from observations and empirical relationships.

Migrated from cyclestrength.pro
"""

import abc
import numba as nb
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline, make_smoothing_spline

from sftpy import simrc as rc

from ..component import Component
from ..util import Timestep
from ..util.funcs import smooth

latlo = rc["cycle.latlo"]
lathi = rc["cycle.lathi"]
period = rc["cycle.period"]
ovr = rc["cycle.ovr"]
peak = rc["cycle.peak"]
mult = rc["cycle.mult"]
loglvl = rc["component.loglvl"]



class Cycle(Component, metaclass=abc.ABCMeta):
    """
    Base class for components describing a stellar magnetic cycle using
    empirical relationships from Schrijver's model.
    """

    prefix = "[cycle]"
    
    def __init__(self,
                 timestep: Timestep,
                 latlo: float=latlo,
                 lathi: float=lathi,
                 period: float=period,
                 ovr: float=ovr,
                 peak: float=peak,
                 mult: float=mult,
                 loglvl: int=loglvl):
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
        """
        Determines the relative strength of the solar activity cycle at the
        current timestep.

        Returns
        -------
        source : float
            Source strength.
        latsource : float
            Source emergence latitude.
        """
        ...



# @nb.jit(cache=True)
def cycle_prescription(time: float,
                       pd: float,
                       ovr: float,
                       peak: float,
                       mult: float,
                       latlo: float,
                       lathi: float):
    """
    Calculates the emergence strength and latitude for the current and
    following cycles according to Schrijver & Title 2001 EQ 2.

    Parameters
    ----------
    time : float
        Current timestep (yr).
    pd : float
        Period of the activity cycle (yr)
    ovr : float
        Overlap between the current and following cycle (yr)
    peak : float
        Peak cycle strength (UNITS??)
    mult : float
        Cycle strength multiplier (dimensionless)
    latlo : float
        Minimum average emergence latitude over the course of the cycle (deg???)
    lathi : float
        Maximum average emergence latitude over the course of the cycle (deg???)

    """

    amax = 2 * np.pi * peak / (pd + 2 * ovr)
    a = np.full(2, amax / peak, dtype=np.float64)
    a[0] *= np.mod(time + pd, pd)
    a[1] *= np.mod(time + pd / 2, pd)

    # 0-year overlap between cycles
    if ovr == 0.0:
        b = 5
        c = 5.81
    # 2-year overlap between cycles
    elif ovr == 2.0:
        b = 8
        c = 8.39
    else:
        b = (amax * np.cos(amax) + np.sin(amax)) / \
                (2 * np.sin(amax) * amax**2) * np.pi**2
        c = amax * np.exp(-(amax / np.pi)**2 * b) / np.pi / \
                max(np.sin(amax), 0)

    # clipped sin
    sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

    # flagged sin
    sin_flag = np.sin(a) > 0

    # ref???
    # source strength calculation per S&T 01 EQ 2
    source = np.array([1, -1]) * sin_clip * mult * a / np.pi * \
            np.exp(-(a / np.pi)**2 * b) * c
    # source emergence latitude follows empirical equatorward linear trend
    latsource = (lathi - (lathi - latlo) * a / np.pi) * sin_flag

    return source, latsource



class CYC0(Cycle):
    """
    Schrijver cycle mode 0: fixed activity level independent of time.
    """

    prefix = "[cycle-0]"

    def cycle(self):
        source = np.array([self._mult], dtype=np.float64)
        latsource = np.array([self._lathi], dtype=np.float64)
        return source, latsource



class CYC1(Cycle):
    """
    Schrijver cycle mode 1: fixed-amplitude activity cycle
    """

    prefix = "[cycle-1]"

    def cycle(self):

        # convert timestep: (s) -> (yr)
        time = self._timestep.gettime() / 86400 / 365

        # calculate source strength and source emergence latitude
        source, latsource = cycle_prescription(time, self._pd, self._ovr, self._peak,
                                               self._mult, self._latlo, self._lathi)

        return source, latsource



class CYC2(Cycle):
    """
    Schrijver cycle mode 2: cycle amplitude modulated according to file specs
    """

    def __init__(self,
                 timestep: Timestep,
                 fname: str,
                 latlo: float=latlo,
                 lathi: float=lathi,
                 period: float=period,
                 ovr: float=ovr,
                 peak: float=peak,
                 mult: float=mult,
                 width: float=10,
                 order: int=3,
                 loglvl: int=loglvl):
        super().__init__(timestep, latlo, lathi, period, ovr, peak, mult,
                         loglvl)
        self._spl = ssn_interp(fname, width, order)

    
    prefix = "[cycle-2]"

    def cycle(self):

        # convert timestep: (s) -> (yr)
        time = self._timestep.gettime() / 86400 / 365

        # calculate source strength and source emergence latitude
        source, latsource = cycle_prescription(time, self._pd, self._ovr, self._peak,
                                               self._mult, self._latlo, self._lathi)

        # use interpolated Sunspot number data to modulate activity
        s_interp = self._spl(time)
        source *= s_interp

        return source, latsource



class CYC3(Cycle):
    """
    Schrijver cycle mode 3: emerging no regions at all.
    """

    prefix = "[cycle-3]"

    def cycle(self):
        source = np.array([0.0], dtype=np.float64)
        latsource = np.array([90.0], dtype=np.float64)
        return source, latsource



class CYC4(Cycle):
    """
    Schrijver cycle mode 4: activity cycle matching solar minima records
    """

    def __init__(self,
                 timestep: Timestep,
                 fname: str,
                 lathi: float=lathi,
                 peak: float=peak,
                 mult: float=mult,
                 width: int=10,
                 order: int=3,
                 loglvl: int=loglvl):
        super().__init__(timestep,
                         latlo=0.0,
                         lathi=lathi,
                         period=21.9,
                         ovr=3.0,
                         peak=peak,
                         mult=mult,
                         loglvl=loglvl)
        self._spl = ssn_interp(fname, width, order)
        self._polarity = polarity = np.ones(len(CYC4.minima), dtype=np.float64)
        polarity[::2] *= -1

    prefix = "[cycle-4]"

    minima = np.array([
        1635.1, 1646.0, 1657.0, 1668.0, 1679.0, 1690.0, 1700.0, 1713.5, 1724.0,
        1733.5, 1745.0, 1756.0, 1767.0, 1775.5, 1784.0, 1798.5, 1811.0, 1825.0,
        1833.5, 1844.0, 1856.5, 1867.0, 1879.0, 1890.0, 1901.4, 1913.4, 1923.8,
        1934.4, 1944.3, 1954.3, 1964.5, 1976.6, 1986.5, 1996.7, 2006.9, 2018.2,
        2029.2, 2040.1]) - 1646.001

    # force solar conditions
    pd = 21.9
    ovr = 3.0
    latlo = 0.0

    # matchsolarrecords
    def cycle(self):

        # time in years
        time = self._timestep.gettime() / 86400 / 365
        minima = CYC4.minima
        polarity = self._polarity

        # determine index of first cycle
        yi = np.nonzero(time < minima)[0][0] - 2
        yd = 2 * np.array([minima[yi+1] - minima[yi],
                           minima[yi+2] - minima[yi+1]])

        a = np.pi * np.array([(time - minima[yi]) / (yd[0] / 2 + self._ovr),
                              (time - minima[yi+1]) / (yd[1] / 2 + self._ovr)])
        # clipped sin
        sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

        # flagged sin
        sin_flag = np.sin(a) > 0

        source = self._mult * sin_clip * a / np.pi * polarity[[yi,yi+1]] * \
                np.exp(-(a / np.pi) ** 2 * 8) * 8.39
        latsource = (self._lathi - (self._lathi - self._latlo) * a / np.pi) * sin_flag

        if polarity[yi] < 0:
            source = np.roll(source, 1)
            latsource = np.roll(latsource, 1)

        s_interp = self._spl(time)
        source *= s_interp

        return source, latsource



def ssn_interp(fname: str, width: int=10, order: int=3) -> BSpline:
    """
    Interpolate sunspot number data

    Parameters
    ----------
    fname : str
        Name of sunspot number data file
    width : float
        Size of boxcar smoothing kernel (yr)
    order : int
        Order of interpolating polynomial
    """
    # load sunspot number data from file
    data = np.load(fname)
    # start spline at year 0
    y = data[:,0] - data[0,0]
    s0 = data[:,1]
    s = np.empty_like(s0)

    # compute cycle strength by taking maximum of every 10 year interval
    for i in range(data.shape[0] - width):
        s[i+width/2] = np.max(s0[i:i+width+1])

    # cycle strength normalized to ~unit strength in 1980's and 1990's
    s = smooth(s, width) / 155.0

    # assume the last 10 years to remain at fixed strength
    s[-11:] = 0.75

    # assume the first 18 years to remain at 0.01
    s[:19] = 0.01

    # cubic spline, not a smoothing interp
    # no way to set IDL kw Sigma = 1.0 (default) so this spline will touch every
    # point instead of being loose/smooth
    # bspl = make_interp_spline(y, s, k=order, bc_type="clamped")
    bspl = make_smoothing_spline(y, s, lam=1.0)
    return bspl
