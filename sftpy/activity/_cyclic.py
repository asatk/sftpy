"""
These components encapsulate the behavior of a stellar activity cycle as
determined from observations and empirical relationships.
"""

import numpy as np
from numpy import typing as npt
from scipy.interpolate import BSpline, make_smoothing_spline

from ..sim import Timeline, Data
from ..util.funcs import smooth

from .constants import CYC_LAT_START, CYC_LAT_END
from .constants import CYC_PERIOD, CYC_OVERLAP, CYC_PEAK, CYC_PHASE_CONST
from .constants import SMOOTH_WIDTH, INTERP_ORDER
from .constants import MULTIPLIER
from ._activity import ScheduledActivity



class SchrijverCycle(ScheduledActivity):
    """
    Base class for components describing a stellar magnetic activity using
    empirical relationships from Schrijver's model.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 lat_start: float = CYC_LAT_START,
                 lat_end: float = CYC_LAT_END,
                 period: float = CYC_PERIOD,
                 overlap: float = CYC_OVERLAP,
                 peak: float = CYC_PEAK,
                 phase_const: float = CYC_PHASE_CONST,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data, multiplier)
        self._lat_start = lat_end
        self._lat_end = lat_start
        self._period = period
        self._overlap = overlap
        self._peak = peak
        self._phase_const = phase_const

        # peak phase of cycle
        amax = 2 * np.pi * peak / (period + 2 * overlap)

        # 0-year overlap between cycles
        if overlap == 0.0:
            b = 5
            c = 5.81
        # 2-year overlap between cycles
        elif overlap == 2.0:
            b = 8
            c = 8.39
        else:
            sin_amax = np.sin(amax)
            b = (amax * np.cos(amax) + sin_amax) / \
                (2 * sin_amax * amax ** 2) * np.pi ** 2
            c = np.exp((amax / np.pi) ** 2 * b) / (
                    amax / np.pi * max(sin_amax, 0))

        def _schedule_strength():
            """
            Calculates the emergence strength and mean latitude for all times
            according to Schrijver & Title 2001 EQ 2. Strengths and latitudes are
            stored as hidden attributes _strength_timeline and _latitude_timeline.
            """

            # convert timestep: (s) -> (yr)
            t_years = time.time_schedule("y")

            # apply cycle phase offset
            t_years += phase_const * period

            # calculate the current phase of the cycle
            phase = np.fmod(t_years, period) / (period + 2 * overlap)

            # schrijver's a parameter
            a = 2 * np.pi * phase

            # argument in schrijver's activity formula
            arg = a / np.pi

            # only includes phases in first half-cycle of full cycle
            sin_clip = np.clip(np.sin(a), a_min=0.0, a_max=None)

            # source strength calculation per S&T 01 EQ 2
            strength_timeline = multiplier * sin_clip * arg * \
                                np.exp(-arg ** 2 * b) * c

            return strength_timeline

        def _schedule_latitude():
            """
            Calculates the emergence strength and mean latitude for all times
            according to Schrijver & Title 2001 EQ 2. Strengths and latitudes are
            stored as hidden attributes _strength_timeline and _latitude_timeline.
            """

            # convert timestep: (s) -> (yr)
            t_years = time.time_schedule("y")

            # apply cycle phase offset
            t_years += phase_const * period

            # calculate the current phase of the cycle
            phase = np.fmod(t_years, period) / (period + 2 * overlap)

            # source emergence latitude follows empirical equatorward linear trend
            latitude_timeline = lat_end - (lat_end - lat_start) * 2 * phase

            return latitude_timeline

        self._get_strength = time.create_schedule(_schedule_strength)
        self._get_latitude = time.create_schedule(_schedule_latitude)

        self._strength_schedule = time.get_schedule(self._get_strength)
        self._latitude_schedule = time.get_schedule(self._get_latitude)

    def strength(self) -> float:
        return self._get_strength()

    def latitude(self) -> float:
        return self._get_latitude()

    def strength_schedule(self) -> npt.NDArray:
        return self._strength_schedule

    def latitude_schedule(self) -> npt.NDArray:
        return self._latitude_schedule



class ModulatedSchrijverCycle(SchrijverCycle):
    """
    Schrijver activity mode 2: activity amplitude modulated according to file specs
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 fname: str,
                 smooth_width: int = SMOOTH_WIDTH,
                 interp_order: int = INTERP_ORDER,
                 **kwargs):

        super().__init__(time, data, **kwargs)

        self._fname = fname
        self._smooth_width = smooth_width
        self._interp_order = interp_order

        spl = ssn_interp(fname, smooth_width, interp_order)

        def _schedule_ssn_modulation():
            # convert timestep: (s) -> (yr)
            t_years = time.time_schedule("y")

            # apply cycle phase offset
            t_years += self._phase_const * self._period

            # use interpolated Sunspot number data to modulate activity
            return spl(t_years)

        self._get_ssn_modulation = time.create_schedule(_schedule_ssn_modulation)
        self._ssn_schedule = time.get_schedule(self._get_ssn_modulation)
        self._strength_schedule = self._ssn_schedule * super().strength_schedule()

    def strength(self):
        return self._get_ssn_modulation() * super().strength()

    def strength_schedule(self):
        return self._strength_schedule



# TODO finish and test this class once interp data are sourced
# Inherit from ModulatedSchrijverCycle above?
class SolarRecordsCycle(ScheduledActivity):
    """
    Schrijver activity mode 4: activity activity matching solar minima records
    """

    minima = np.array([
        1635.1, 1646.0, 1657.0, 1668.0, 1679.0, 1690.0, 1700.0, 1713.5, 1724.0,
        1733.5, 1745.0, 1756.0, 1767.0, 1775.5, 1784.0, 1798.5, 1811.0, 1825.0,
        1833.5, 1844.0, 1856.5, 1867.0, 1879.0, 1890.0, 1901.4, 1913.4, 1923.8,
        1934.4, 1944.3, 1954.3, 1964.5, 1976.6, 1986.5, 1996.7, 2006.9, 2018.2,
        2029.2, 2040.1]) - 1646.001

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 fname: str,
                 smooth_width: int = SMOOTH_WIDTH,
                 interp_order: int = INTERP_ORDER,
                 lat_start: float = CYC_LAT_START,
                 lat_end: float = CYC_LAT_END,
                 period: float = CYC_PERIOD,
                 overlap: float = CYC_OVERLAP,
                 peak: float = CYC_PEAK,
                 multiplier: float = MULTIPLIER):
        super().__init__(time, data, multiplier)
        self._lat_end = lat_end
        self._lat_start = lat_start
        self._period = period
        self._overlap = 3.0
        self._peak = peak
        self._spl = ssn_interp(fname, smooth_width, interp_order)
        # TODO check this line?
        self._polarity = polarity = np.ones(len(SolarRecordsCycle.minima), dtype=np.float64)
        polarity[::2] *= -1

    # matchsolarrecords
    def strength(self):

        # time in years
        t_years = self._time.time("y")
        minima = SolarRecordsCycle.minima
        polarity = self._polarity

        # determine index of first activity
        yi = np.nonzero(t_years < minima)[0][0] - 2
        yd = 2 * np.array([minima[yi+1] - minima[yi],
                           minima[yi+2] - minima[yi+1]])

        a = np.pi * np.array([(t_years - minima[yi]) / (yd[0] / 2 + self._overlap),
                              (t_years - minima[yi+1]) / (yd[1] / 2 + self._overlap)])
        # clipped sin
        sin_clip = np.clip(np.sin(a), a_min=0, a_max=None)

        # flagged sin
        sin_flag = np.sin(a) > 0

        source = self._multiplier * sin_clip * a / np.pi * \
                np.exp(-(a / np.pi) ** 2 * 8) * 8.39
        source *= polarity[[yi,yi+1]]
        latsource = (self._lat_start - (self._lat_start - self._lat_end) * a / np.pi) * sin_flag

        if polarity[yi] < 0:
            source = np.roll(source, 1)
            latsource = np.roll(latsource, 1)

        s_interp = self._spl(t_years)
        source *= s_interp

        return source

    def latitude(self) -> float:

        # time in years
        t_years = self._time.time("y")
        minima = SolarRecordsCycle.minima
        polarity = self._polarity

        # determine index of first activity
        yi = np.nonzero(t_years < minima)[0][0] - 2
        yd = 2 * np.array([minima[yi + 1] - minima[yi],
                           minima[yi + 2] - minima[yi + 1]])

        a = np.pi * np.array([(t_years - minima[yi]) / (yd[0] / 2 + self._overlap),
                              (t_years - minima[yi + 1]) / (
                                          yd[1] / 2 + self._overlap)])

        # flagged sin
        sin_flag = np.sin(a) > 0


        latsource = (self._lat_start - (self._lat_start - self._lat_end) * a / np.pi) * sin_flag
        return latsource

    def strength_schedule(self) -> npt.NDArray[np.float64]:
        return 1.

    def latitude_schedule(self) -> npt.NDArray[np.float64]:
        return 0.0



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
    # TODO decide how to get/update sunspot number data
    # load sunspot number data from SIDC or NOAA
    data = np.load(fname)
    # start spline at year 0
    y = data[:,0] - data[0,0]
    s0 = data[:,1]
    s = np.empty_like(s0)

    # compute activity strength by taking maximum of every 10 year interval
    for i in range(data.shape[0] - width):
        s[i+width/2] = np.max(s0[i:i+width+1])

    # activity strength normalized to ~unit strength in 1980's and 1990's
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
