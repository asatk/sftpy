import numpy as np

from ..activity import Activity, ScheduledActivity
from ..sim import Data, rng, Timeline
from ..sim.constants import BINFLUX, STAR_RADIUS
from ..util import schrijver_rv

from .constants import AR_BFIELD, BP_FLUX_POWER_ACT, BP_FLUX_POWER_EPH
from .constants import BP_LAT_WIDTH, BP_LAT_FLUX_FOLD
from .constants import BP_TILT_MEAN, BP_TILT_WIDTH, BP_TILT_FLUX_FOLD, BP_TILT_CONST
from .constants import BP_FLUX_MIN, BP_FLUX_MAX
from .constants import BP_FRACTION_TURBULENT
from .constants import BP_FLAG_NOISE
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
                 tilt_mean: float = BP_TILT_MEAN,
                 tilt_width: float = BP_TILT_WIDTH,
                 tilt_flux_fold: float = BP_TILT_FLUX_FOLD,
                 tilt_const: float = BP_TILT_CONST,
                 flag_noise: bool = BP_FLAG_NOISE):

        super().__init__(time, data, activity)

        self._pwr = pwr
        self._flux_min = flux_min
        self._flux_max = flux_max
        self._ar_bfield = ar_bfield

        self._lat_width = lat_width
        self._lat_flux_fold = lat_flux_fold

        self._tilt_mean = tilt_mean
        self._tilt_width = tilt_width
        self._tilt_flux_fold = tilt_flux_fold
        self._tilt_const = tilt_const

        self._flag_noise = flag_noise

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
        width = self._lat_width * (
                    np.exp(-aflux * BINFLUX / self._lat_flux_fold) + 0.15)
        theta += rng.normal(size=nnew) * width * np.pi / 180
        # latitude -> co-latitude
        theta = np.fmod(np.pi / 2 - theta, np.pi)

        data_new.phi[:nnew] = phi
        data_new.theta[:nnew] = theta
        data_new.flux[:nnew] = flux
        data_new.nflux = nnew

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
            -BINFLUX * aflux / self._tilt_flux_fold) + self._tilt_const
        orient = (rng.normal(size=nnew) * width + self._tilt_mean) * np.pi / 180
        # flip sign for opposite polarity regions in different hemispheres (Hale's Law)
        hemi = np.sign(np.pi / 2 - theta)
        orient = np.pi * (1 - hemi) / 2 + hemi * orient
        # invert polarity based on phase of activity
        orient += np.pi * (strength < 0)

        # Step 4 --- position concentrations
        r = (np.sqrt(flux * BINFLUX * 1e18 / self._ar_bfield / np.pi) + 7e8) / 7e10

        # impose minimum separation of ~0.5 supergranulation of 18Mm
        sep = np.clip(r, a_min=9000 / STAR_RADIUS / 2, a_max=None)

        # number of new concentrations that contain 15e18 Mx w/ at least
        # three equal concentrations per polarity
        percon = np.clip(flux / 3, a_min=1, a_max=None)
        percon = np.trunc(percon)
        percon[flux > (3 * 15 / BINFLUX)] = 15 / BINFLUX

        bulk = np.clip(np.astype(
            flux / percon, np.int64),
            a_min=1, a_max=None)

        rest = np.clip(flux - percon * bulk, a_min=0, a_max=None)

        nadd = bulk + (rest > 0)
        nadd_tot = np.sum(nadd)
        ind_rest = np.cumsum(nadd)[rest > 0] - 1

        r_nadd = np.repeat(r, nadd)
        sep_nadd = np.repeat(sep, nadd)
        percon_nadd = np.repeat(percon, nadd)
        percon_nadd[ind_rest] = rest[rest > 0]

        # one polarity
        offset1 = rng.uniform(high=r_nadd)
        angle1 = rng.uniform(high=2 * np.pi, size=nadd_tot)

        # opposite polarity
        offset2 = rng.uniform(high=r_nadd)
        angle2 = rng.uniform(high=2 * np.pi, size=nadd_tot)

        x_tmp = np.concatenate([
            +sep_nadd + offset1 * np.cos(angle1),
            -sep_nadd + offset2 * np.cos(angle2)
        ])
        y_tmp = np.concatenate([
            offset1 * np.sin(angle1),
            offset2 * np.sin(angle2)
        ])

        # orientation of bipolar spot
        # TODO better way to double these?
        orient_nadd_half = np.repeat(orient, nadd)
        orient_nadd = np.concatenate([orient_nadd_half, orient_nadd_half])

        orient_tmp = orient_nadd + np.pi / 2
        coso = np.cos(orient_tmp)
        sino = np.sin(orient_tmp)
        xo = coso * x_tmp + sino * y_tmp
        yo = -sino * x_tmp + coso * y_tmp

        # location of bipolar active region / concentration
        phi_nadd_half = np.repeat(phi, nadd)
        phi_nadd = np.concatenate([phi_nadd_half, phi_nadd_half])
        theta_nadd_half = np.repeat(theta, nadd)
        theta_nadd = np.concatenate([theta_nadd_half, theta_nadd_half])

        cosphi = np.cos(phi_nadd)
        sinphi = np.sin(phi_nadd)
        costheta = np.cos(theta_nadd)
        sintheta = np.sin(theta_nadd)

        # Cartesian coordinates of spots
        x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
        y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
        z = costheta - xo * sintheta

        # spherical coordinates of spots
        aphi = np.fmod(np.arctan2(y, x), 2 * np.pi)
        atheta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))

        # TODO move to emerge? how to do this w/ net zero flux
        # Poisson noise added to each concentration
        if self._flag_noise:
            scale_nadd = np.sqrt(percon_nadd)
            noise = rng.normal(size=nadd_tot) * scale_nadd
            noise[ind_rest] = 0.0

            percon_nadd += noise
            percon_nadd[ind_rest] -= noise[ind_rest]

        # add both polarities of spots
        aflux = np.concatenate([
            +percon_nadd,
            -percon_nadd
        ]).astype(np.int64)

        data_full = self._data_full
        data_full.clear()
        data_full.append_sources(aphi, atheta, aflux, 2 * nadd_tot)

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
                 tilt_const: float = BP_TILT_CONST,
                 flag_noise: bool = BP_FLAG_NOISE):

        super().__init__(time, data, activity, pwr, flux_min, flux_max, ar_bfield,
                         lat_width, lat_flux_fold, tilt_mean, tilt_width,
                         tilt_flux_fold, tilt_const, flag_noise)

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
            a = 8 * strength_abs ** (1 / 3) * fraction_trb + (1 - fraction_trb)

            nnew = a * factor * dt_func("d")
            frac = nnew - np.trunc(nnew)
            remainder = rng.uniform(size=time.nstep) < frac
            nnew_int = np.astype(nnew, np.int64) + remainder

            return nnew_int

        if isinstance(activity, ScheduledActivity):
            self._get_nnew = time.create_schedule(calculate_nnew)
        else:
            self._get_nnew = calculate_nnew