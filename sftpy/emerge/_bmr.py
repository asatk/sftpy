import numpy as np

from ..component import Component
from ..util import schrijver_rv
from ..util import powerlaw_rv



class BipoleRegion(Component):

    prefix = "[bipole]"

    def __init__(self,
                 p: float,
                 minflux: float,
                 maxflux: float,
                 avefluxd: float,
                 dt: float,
                 turbulent: float,
                 lat_width: float,
                 lat_fold: float,
                 joy: float,
                 joy_width: float,
                 joy_fold: float,
                 sjzero: float,
                 rad: float,
                 binflux: float,
                 rng: np.random.Generator,
                 loglvl: int=0
                 ):
        super().__init__(loglvl)
        self._p = p
        self._minflux = minflux
        self._maxflux = maxflux
        self._avefluxd = avefluxd
        self._dt = dt
        self._turbulent = turbulent

        self._init_flux_ar()
        self._init_flux_eph()

        self._lat_width = lat_width
        self._lat_fold = lat_fold

        self._joy = joy
        self._joy_width = joy_width
        self._joy_fold = joy_fold
        self._sjzero = sjzero

        self._rad = rad

        self._binflux = binflux
        self._rng = rng

    def _init_flux_ar(self):
        pm1 = self._p - 1
        scale = (1.5 * self._avefluxd) ** pm1 / -pm1
        rangefactor = self._maxflux ** -pm1 - self._minflux ** -pm1
        self._ntotalfactor_ar = 2 * self._dt / 86400 * scale * rangefactor

    def _init_flux_eph(self):
        p = self._p + 1
        pm1 = p - 1
        scale = (1.5 * self._avefluxd) ** pm1 / -pm1
        rangefactor = self._maxflux ** -pm1 - self._minflux ** -pm1
        self._ntotalfactor_eph = 2 * self._dt / 86400 * scale * rangefactor


    def sample_flux(self, source: float):
        rng = self._rng

        source = np.abs(source)

        ## [1] High-flux tail dominant for large regions
        a = 8 * source
        ntotal_ar = a * self._ntotalfactor_ar
        frac_ar = ntotal_ar - int(ntotal_ar)
        ntotal_ar = int(ntotal_ar) + (rng.uniform() < frac_ar)

        # rv_ar = powerlaw_rv(ntotal_ar, -self._p, self._minflux / 2 / binflux,
        #                     self._maxflux / 2 / binflux, rng)
        rv_ar = schrijver_rv(ntotal_ar, self._p, self._minflux / 2 / self._binflux,
                              self._maxflux / 2 / self._binflux, rng)
        flux_ar = np.astype(rv_ar, np.int64)

        ## [2] Low-flux tail dominated by ephemeral regions
        a = 8 * source ** (1 / 3) * self._turbulent + (1 - self._turbulent)
        ntotal_eph = a * self._ntotalfactor_eph
        frac_eph = ntotal_eph - int(ntotal_eph)
        ntotal_eph = int(ntotal_eph) + (self._rng.uniform() < frac_eph)

        # rv_eph = powerlaw_rv(ntotal_eph, -self._p + 1,self._minflux / 2 / binflux,
        #                      self._maxflux / 2 / binflux, rng)
        rv_eph = schrijver_rv(ntotal_eph, self._p + 1, self._minflux / 2 / self._binflux,
                           self._maxflux / 2 / self._binflux, rng)
        flux_eph = np.astype(rv_eph, np.int64)

        self.log(1,
                 f"Active = {ntotal_ar}\t" + \
                 f"Ephemeral = {ntotal_eph}\t" + \
                 f"All = {ntotal_ar + ntotal_eph}")

        flux = np.r_[flux_ar, flux_eph]
        return flux

    def sample_phi(self, ntotal: int) -> np.ndarray:
        phi = self._rng.uniform(high=2*np.pi, size=ntotal)
        return phi

    def sample_theta(self, latsource: float, flux: np.ndarray, ntotal: int) -> np.ndarray:
        rng = self._rng

        theta = latsource * np.pi / 180 * rng.choice([-1, 1], size=ntotal)
        width = self._lat_width * (np.exp(-flux * self._binflux / self._lat_fold) + 0.15)
        theta += rng.normal(scale=width * np.pi / 180, size=ntotal)
        # TODO introduced this myself just to prevent stuff from going oob
        # theta = np.clip(theta, a_min=-np.pi/2, a_max=np.pi/2)
        # latitude -> co-latitude
        theta = (np.pi / 2 - theta) % np.pi
        return theta

    def sample_tilt(self, source: float, theta: np.ndarray, flux: np.ndarray, ntotal) -> np.ndarray:
        width = self._joy_width * np.exp(-self._binflux * flux / self._joy_fold) + self._sjzero
        orient = self._rng.normal(loc=self._joy, scale=width, size=ntotal) * np.pi / 180
        # flip sign for opposite polarity regions in different hemispheres (Hale's Law)
        hemi = np.sign(np.pi / 2 - theta)
        orient = np.pi * (1 - hemi) / 2 + hemi * orient
        # invert polarity based on phase of cycle
        orient += np.pi * (source < 0)
        return orient

    def make_concentrations(self, phi, theta, flux, orient):
        rng = self._rng

        r = (np.sqrt(flux * self._binflux * 1e18 / self._avefluxd / np.pi) + 7e8) / 7e10
        # impose minimum separation of ~0.5 supergranulation of 18Mm
        sep = np.clip(r, a_min=9000 / self._rad / 2, a_max=None)
        # number of new concentrations that contain 15e18 Mx w/ at least
        # three equal concentrations per polarity
        percon = np.clip(flux / 3., a_min=1, a_max=None).astype(int)
        percon[percon > (15. / self._binflux)] = 15. / self._binflux

        self.log(1, f"percon = {np.sum(percon)}")

        # bulk = np.clip(flux // percon, a_min=1, a_max=None)
        bulk = np.clip(np.astype(
            flux / percon, np.int64),
            a_min=1, a_max=None)

        self.log(1, f"bulk = {np.mean(bulk)}")

        rest = np.clip(flux - percon * bulk, a_min=0, a_max=None)

        # self.log(1, f"rest = {np.sum(rest)}")
        self.log(1, f"rest = {np.count_nonzero(rest)}")

        nadd = bulk + (rest > 0)
        nadd[flux < bulk * percon] = 1
        ind_rest = np.cumsum(nadd) - 1

        r_nadd = np.repeat(r, nadd)
        sep_nadd = np.repeat(sep, nadd)
        percon_nadd = np.repeat(percon, nadd)
        percon_nadd[ind_rest] = rest
        nadd_tot = ind_rest[-1] + 1

        # one polarity
        offset1 = rng.uniform(high=r_nadd)
        angle1 = rng.uniform(high=2 * np.pi, size=nadd_tot)

        # opposite polarity
        offset2 = rng.uniform(high=r_nadd)
        angle2 = rng.uniform(high=2 * np.pi, size=nadd_tot)

        x_tmp = np.r_[sep_nadd + offset1 * np.cos(angle1),
                      -sep_nadd + offset2 * np.cos(angle2)]
        y_tmp = np.r_[offset1 * np.sin(angle1),
                      offset2 * np.sin(angle2)]

        # orientation of bipolar spot
        # TODO better way to double these?
        orient_nadd_half = np.repeat(orient, nadd)
        orient_nadd = np.r_[orient_nadd_half, orient_nadd_half]

        orient_tmp = orient_nadd + np.pi / 2
        coso = np.cos(orient_tmp)
        sino = np.sin(orient_tmp)
        xo = coso * x_tmp + sino * y_tmp
        yo = -sino * x_tmp + coso * y_tmp

        # location of bipolar active region / concentration
        phi_nadd_half = np.repeat(phi, nadd)
        phi_nadd = np.r_[phi_nadd_half, phi_nadd_half]
        theta_nadd_half = np.repeat(theta, nadd)
        theta_nadd = np.r_[theta_nadd_half, theta_nadd_half]

        cosphi = np.cos(phi_nadd)
        sinphi = np.sin(phi_nadd)
        costheta = np.cos(theta_nadd)
        sintheta = np.sin(theta_nadd)

        # Cartesian coordinates of spots
        x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
        y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
        z = costheta - xo * sintheta

        # spherical coordinates of spots
        aphi = np.arctan2(y, x) % (2 * np.pi)
        atheta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))

        # Poisson noise added to each concentration
        scale_nadd = np.sqrt(percon_nadd)
        noise = rng.normal(scale=scale_nadd)

        # add both polarities of spots
        aflux = np.r_[percon_nadd + noise, -percon_nadd - noise]

        # IDL code has remainder concentration w/o noise...
        aflux[ind_rest] = aflux[ind_rest] - noise[ind_rest]
        aflux[ind_rest + nadd_tot] = aflux[ind_rest + nadd_tot] + noise[
            ind_rest]
        aflux = np.astype(aflux, np.int64)

        return aphi, atheta, aflux

