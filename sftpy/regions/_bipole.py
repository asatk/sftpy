import numpy as np

from ..activity import Activity, ScheduledActivity
from ..sim import Data, rng, Timeline
from ..sim.constants import BINFLUX, STAR_RADIUS
from ..util import schrijver_rv

from .constants import AR_BFIELD, BP_FLUX_POWER_ACT, BP_FLUX_POWER_EPH
from .constants import BP_LAT_WIDTH, BP_LAT_FLUX_FOLD, BP_LAT_CONST
from .constants import BP_TILT_MEAN, BP_TILT_WIDTH, BP_TILT_FLUX_FOLD, BP_TILT_CONST
from .constants import BP_FLUX_MIN, BP_FLUX_MAX, BP_FRACTION_TURBULENT
from ._magneticregion import MagneticRegion



class SchrijverBipole(MagneticRegion):

    ### [1] High-flux tail dominant for (large) active regions

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 activity: Activity,
                 pwr: float = BP_FLUX_POWER_ACT,
                 flux_min: int = BP_FLUX_MIN,
                 flux_max: int = BP_FLUX_MAX,
                 ar_bfield: float = AR_BFIELD,
                 lat_width: float = BP_LAT_WIDTH,
                 lat_flux_fold: float = BP_LAT_FLUX_FOLD,
                 lat_const: float = BP_LAT_CONST,
                 tilt_mean: float = BP_TILT_MEAN,
                 tilt_width: float = BP_TILT_WIDTH,
                 tilt_flux_fold: float = BP_TILT_FLUX_FOLD,
                 tilt_const: float = BP_TILT_CONST):

        super().__init__(time, data, activity)

        self._pwr = pwr
        self._flux_min = flux_min
        self._flux_max = flux_max
        self._ar_bfield = ar_bfield

        self._lat_width = lat_width
        self._lat_flux_fold = lat_flux_fold
        self._lat_const = lat_const

        self._tilt_mean = tilt_mean
        self._tilt_width = tilt_width
        self._tilt_flux_fold = tilt_flux_fold
        self._tilt_const = tilt_const

        if isinstance(activity, ScheduledActivity):
            strength_func = activity.strength_schedule
            dt_func = time.dt_schedule
        else:
            strength_func = activity.strength
            dt_func = time.dt

        def calculate_nnew():
            strength_abs = np.abs(strength_func())
            pm1 = pwr - 1
            scale = (1.5 * ar_bfield) ** pm1 / -pm1
            range_factor = flux_max ** -pm1 - flux_min ** -pm1
            factor = 2 * scale * range_factor
            a = 8 * strength_abs

            nnew = a * factor * dt_func("d")
            frac = nnew - np.trunc(nnew)
            remainder = rng.uniform(size=time.nstep) < frac
            nnew_int = np.astype(nnew, np.int64) + remainder

            return nnew_int

        if isinstance(activity, ScheduledActivity):
            self._get_nnew = time.create_schedule(calculate_nnew)
        else:
            self._get_nnew = calculate_nnew

        self._data_new = Data()
        self._data_full = Data()



    def place_new_spots(self) -> Data:

        nnew = self._get_nnew()
        data_new = self._data_new

        data_new.clear()

        # nothing to emerge
        if nnew == 0:
            return data_new

        # Step 1 --- determine bipole size distribution

        # TODO replace with real power law sampler not replacement sampling
        rv = schrijver_rv(nnew,
                          self._pwr,
                          self._flux_min / 2,
                          self._flux_max,
                          rng)
        flux = np.astype(rv / BINFLUX, np.int64)

        # Step 2 --- determine positions

        phi = rng.uniform(high=2 * np.pi, size=nnew)

        latitude = self.activity.latitude()
        aflux = np.abs(flux)
        theta = latitude * np.pi / 180 * rng.choice([-1, 1], size=nnew)
        width = self._lat_width * np.exp(
            -aflux * BINFLUX / self._lat_flux_fold) + self._lat_const
        theta += rng.normal(size=nnew) * width * np.pi / 180
        # latitude -> co-latitude
        theta = np.fmod(np.pi / 2 - theta, np.pi)

        data_new.insert_sources(phi, theta, flux, nnew)

        return data_new



    def fill_new_spots(self, data: Data):

        nnew = data.nflux
        phi = data.phi[:nnew]
        theta = data.theta[:nnew]
        flux = data.flux[:nnew]

        # Step 3 --- orientation of bipole axes
        strength = self.activity.strength()
        aflux = np.abs(flux)
        width = self._tilt_width * np.exp(
            -aflux * BINFLUX / self._tilt_flux_fold) + self._tilt_const
        orient = (rng.normal(size=nnew) * width + self._tilt_mean) * np.pi / 180
        # flip sign for opposite polarity regions in different hemispheres (Hale's Law)
        hemi = np.sign(np.pi / 2 - theta)
        orient = np.pi * (1 - hemi) / 2 + hemi * orient
        # invert polarity based on phase of activity
        orient += np.pi * (strength < 0)

        # Step 4 --- position concentrations
        # TODO determine where these 7eN hard-coded values come from...
        # assign to constants if necessary. is this that 7km min sep??
        r = (np.sqrt(flux * BINFLUX * 1e18 / self._ar_bfield / np.pi) + 7e8) / 7e10

        # impose minimum separation of ~0.5 supergranulation of 18Mm
        sep = np.clip(r, a_min=9000 / STAR_RADIUS / 2, a_max=None)

        # number of new concentrations that contain 15e18 Mx w/ at least
        # three equal concentrations per polarity
        percon = np.clip(flux / 3, a_min=1, a_max=None)
        percon[flux > (3 * 15 / BINFLUX)] = 15 / BINFLUX
        percon = percon.astype(np.int64)

        bulk = np.clip(np.astype(
            flux / percon, np.int64),
            a_min=1, a_max=None)

        rest = np.clip(flux - percon * bulk, a_min=0, a_max=None)

        nadd = bulk + (rest > 0)
        nadd_tot = np.sum(nadd)

        r_nadd = np.repeat(r, nadd)
        sep_nadd = np.repeat(sep, nadd)
        percon_nadd = np.repeat(percon, nadd)

        # flux of concentrations in each pole
        flux1 = +percon_nadd
        flux2 = -percon_nadd

        # one polarity
        offset1 = rng.uniform(high=r_nadd)
        angle1 = rng.uniform(high=2 * np.pi, size=nadd_tot)

        # opposite polarity
        offset2 = rng.uniform(high=r_nadd)
        angle2 = rng.uniform(high=2 * np.pi, size=nadd_tot)

        # orientation of bipolar spot
        orient_nadd = np.repeat(orient, nadd) + np.pi / 2

        coso = np.cos(orient_nadd)
        sino = np.sin(orient_nadd)

        # location of bipolar active region / concentration
        phi_nadd = np.repeat(phi, nadd)
        theta_nadd = np.repeat(theta, nadd)

        cosphi = np.cos(phi_nadd)
        sinphi = np.sin(phi_nadd)
        costheta = np.cos(theta_nadd)
        sintheta = np.sin(theta_nadd)

        #
        # transform concentrations in (+) pole to points in a circle on a sphere
        #

        # separate (+) pole from region center according to sampled separation
        x1 = +sep_nadd + offset1 * np.cos(angle1)
        y1 = offset1 * np.sin(angle1)

        # tilt region according to sampled orientation
        x1tilt = +coso * x1 + sino * y1
        y1tilt = -sino * x1 + coso * y1

        # Cartesian coordinates of (+) pole's circle on the sphere
        x1cart = 1.0    * cosphi * sintheta + \
                 x1tilt * cosphi * costheta - \
                 y1tilt * sinphi
        y1cart = 1.0    * sinphi * sintheta + \
                 x1tilt * sinphi * costheta + \
                 y1tilt * cosphi
        z1cart = 1.0    * costheta - \
                 x1tilt * sintheta

        # Cartesian coordinates to spherical coordinates
        phi1 = np.fmod(np.arctan2(y1cart, x1cart), 2 * np.pi)
        theta1 = np.arccos(
            z1cart / np.sqrt(x1cart ** 2 + y1cart ** 2 + z1cart ** 2))

        #
        # transform concentrations in (-) pole to points in a circle on a sphere
        #

        # separate (-) pole from region center according to sampled separation
        x2 = -sep_nadd + offset2 * np.cos(angle2)
        y2 = offset2 * np.sin(angle2)

        # tilt region according to sampled orientation
        x2tilt = +coso * x2 + sino * y2
        y2tilt = -sino * x2 + coso * y2

        # Cartesian coordinates of (-) pole's circle on the sphere
        x2cart = 1.0    * cosphi * sintheta + \
                 x2tilt * cosphi * costheta - \
                 y2tilt * sinphi
        y2cart = 1.0    * sinphi * sintheta + \
                 x2tilt * sinphi * costheta + \
                 y2tilt * cosphi
        z2cart = 1.0    * costheta - \
                 x2tilt * sintheta

        # Cartesian to spherical coordinates
        phi2 = np.fmod(np.arctan2(y2cart, x2cart), 2 * np.pi)
        theta2 = np.arccos(
            z2cart / np.sqrt(x2cart ** 2 + y2cart ** 2 + z2cart ** 2))

        # add both poles of region to final list of concentrations
        phi_add = np.concatenate([phi1, phi2])
        theta_add = np.concatenate([theta1, theta2])
        flux_add = np.concatenate([flux1, flux2])

        # return list of all concentrations in bipole regions to be emerged
        data_full = self._data_full
        data_full.insert_sources(phi_add, theta_add, flux_add, 2 * nadd_tot)
        return data_full



class SchrijverEphemeral(SchrijverBipole):

    ### [2] Low-flux tail dominated by ephemeral regions

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 activity: Activity,
                 fraction_trb: float = BP_FRACTION_TURBULENT,
                 pwr: float = BP_FLUX_POWER_EPH,
                 flux_min: int = BP_FLUX_MIN,
                 flux_max: int = BP_FLUX_MAX,
                 ar_bfield: float = AR_BFIELD,
                 lat_width: float = BP_LAT_WIDTH,
                 lat_flux_fold: float = BP_LAT_FLUX_FOLD,
                 tilt_mean: float = BP_TILT_MEAN,
                 tilt_width: float = BP_TILT_WIDTH,
                 tilt_flux_fold: float = BP_TILT_FLUX_FOLD,
                 tilt_const: float = BP_TILT_CONST):

        super().__init__(time, data, activity, pwr, flux_min, flux_max, ar_bfield,
                         lat_width, lat_flux_fold, tilt_mean, tilt_width,
                         tilt_flux_fold, tilt_const)

        self._fraction_trb = fraction_trb

        if isinstance(activity, ScheduledActivity):
            strength_func = activity.strength_schedule
            dt_func = time.dt_schedule

        else:
            strength_func = activity.strength
            dt_func = time.dt

        def calculate_nnew():
            strength_abs = np.abs(strength_func())
            pm1 = pwr - 1
            scale = (1.5 * ar_bfield) ** pm1 / -pm1
            range_factor = flux_max ** -pm1 - flux_min ** -pm1
            factor = 2 * scale * range_factor
            a = 8 * (strength_abs ** (1 / 3) * fraction_trb + (1 - fraction_trb))

            nnew = a * factor * dt_func("d")
            frac = nnew - np.trunc(nnew)
            remainder = rng.uniform(size=time.nstep) < frac
            nnew_int = np.astype(nnew, np.int64) + remainder

            return nnew_int

        if isinstance(activity, ScheduledActivity):
            self._get_nnew = time.create_schedule(calculate_nnew)
        else:
            self._get_nnew = calculate_nnew