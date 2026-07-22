import numpy as np

from sftpy import rng



from ...util import schrijver_rv
from ...util import powerlaw_rv

from ._magneticregion import MagneticRegion



class BipoleRegion(MagneticRegion):

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
                 mode_ar: int=1,
                 mode_eph: int=1,
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

        self._mode_ar = mode_ar
        self._mode_eph = mode_eph



    @property
    def mode_ar(self):
        return self._mode_ar

    @mode_ar.setter
    def mode_ar(self, value):
        self._mode_ar = value

    def toggle_mode_ar(self):
        self._mode_ar = not self._mode_ar

    @property
    def mode_eph(self):
        return self._mode_eph

    @mode_eph.setter
    def mode_eph(self, value):
        self._mode_eph = value

    def toggle_mode_eph(self):
        self._mode_eph = not self._mode_eph



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
        source = np.abs(source)

        ## [1] High-flux tail dominant for large regions
        if self._mode_ar:
            a = 8 * source
            ntotal_ar = a * self._ntotalfactor_ar
            frac_ar = ntotal_ar - int(ntotal_ar)
            ntotal_ar = int(ntotal_ar) + (rng.uniform() < frac_ar)

            # rv_ar = powerlaw_rv(ntotal_ar, -self._p, self._minflux / 2 / binflux,
            #                     self._maxflux / 2 / binflux, rng)
            rv_ar = schrijver_rv(ntotal_ar, self._p, self._minflux / 2,
                                  self._maxflux, rng)
            flux_ar = np.astype(rv_ar / self._binflux, np.int64)
        else:
            ntotal_ar = 0
            flux_ar = np.zeros(0)

        ## [2] Low-flux tail dominated by ephemeral regions
        if self._mode_eph:
            a = 8 * (source ** (1 / 3) * self._turbulent + (1 - self._turbulent))
            ntotal_eph = a * self._ntotalfactor_eph
            frac_eph = ntotal_eph - int(ntotal_eph)
            ntotal_eph = int(ntotal_eph) + (rng.uniform() < frac_eph)

            # TODO check exp of -p + 1 or -p - 1
            # rv_eph = powerlaw_rv(ntotal_eph, -self._p + 1,self._minflux / 2 / binflux,
            #                      self._maxflux / 2 / binflux, rng)
            rv_eph = schrijver_rv(ntotal_eph, self._p + 1, self._minflux / 2,
                               self._maxflux, rng)
            flux_eph = np.astype(rv_eph / self._binflux, np.int64)

        else:
            ntotal_eph = 0
            flux_eph = np.zeros(0)

        self.log(1,
                 f"Cycle Strength = {source:.5f}\t"
                 f"Active = {ntotal_ar}\t" + \
                 f"Ephemeral = {ntotal_eph}\t")

        flux = np.r_[flux_ar, flux_eph]

        # accelerated time mode -- include only regions larger than 2sq deg
        # or 2 * 1.5e18 & avefluxd = 3 avefluxd units of 10^18 Mx/m^2
        # IDL model behavior includes all and only ephemeral regions if
        # cycle source strength relative to Sun is negative

        # fast forward stuff from old model
        # only emerge active regions
        if self._mode_ar and not self._mode_eph:
            ind_big = np.nonzero(flux > (3 * self._avefluxd / self._binflux))[0]
            if len(ind_big) == 0:
                return np.empty(0)
            flux = flux[ind_big]

        # old test mode for negative cycle mult which should be a flag; alas
        # only emerge ephemeral regions
        if not self._mode_ar and self._mode_eph:
            ind_small = np.nonzero(flux < (3 * self._avefluxd / self._binflux))[0]
            if len(ind_small) == 0:
                return np.empty(0)
            flux = flux[ind_small]

        return flux



    def sample_phi(self,
                   flux: np.ndarray,
                   ntotal: int) -> np.ndarray:
        phi = rng.uniform(high=2*np.pi, size=ntotal)
        return phi



    def sample_theta(self,
                     flux: np.ndarray,
                     latsource: float,
                     ntotal: int) -> np.ndarray:

        theta = latsource * np.pi / 180 * rng.choice([-1, 1], size=ntotal)
        width = self._lat_width * (np.exp(-flux * self._binflux / self._lat_fold) + 0.15)
        theta += rng.normal(scale=width * np.pi / 180, size=ntotal)
        # TODO introduced this myself just to prevent stuff from going oob
        # theta = np.clip(theta, a_min=-np.pi/2, a_max=np.pi/2)
        # latitude -> co-latitude
        theta = (np.pi / 2 - theta) % np.pi
        return theta



    def sample_orientation(self,
                           phi: np.ndarray,
                           theta: np.ndarray,
                           flux: np.ndarray,
                           source: float,
                           ntotal: int) -> np.ndarray:
        width = self._joy_width * np.exp(-self._binflux * flux / self._joy_fold) + self._sjzero
        orient = rng.normal(loc=self._joy, scale=width, size=ntotal) * np.pi / 180
        # flip sign for opposite polarity regions in different hemispheres (Hale's Law)
        hemi = np.sign(np.pi / 2 - theta)
        orient = np.pi * (1 - hemi) / 2 + hemi * orient
        # invert polarity based on phase of cycle
        orient += np.pi * (source < 0)
        return orient



    def make_concentrations(self,
                            phi: np.ndarray,
                            theta: np.ndarray,
                            flux: np.ndarray,
                            orient: np.ndarray):
        r = (np.sqrt(flux * self._binflux * 1e18 / self._avefluxd / np.pi) + 7e8) / 7e10
        # impose minimum separation of ~0.5 supergranulation of 18Mm
        sep = np.clip(r, a_min=9000 / self._rad / 2, a_max=None)
        # number of new concentrations that contain 15e18 Mx w/ at least
        # three equal concentrations per polarity
        percon = np.clip(flux / 3., a_min=1, a_max=None).astype(int)
        percon[percon > (15. / self._binflux)] = 15. / self._binflux

        # self.log(0, f"percon = {np.sum(percon)}")

        # bulk = np.clip(flux // percon, a_min=1, a_max=None)
        bulk = np.clip(np.astype(
            flux / percon, np.int64),
            a_min=1, a_max=None)

        # self.log(0, f"bulk = {np.mean(bulk)}")

        rest = np.clip(flux - percon * bulk, a_min=0, a_max=None)

        # self.log(2, f"rest = {np.sum(rest)}")
        # self.log(0, f"rest = {np.count_nonzero(rest)}")

        nadd = bulk + (rest > 0)
        nadd[flux < bulk * percon] = 1
        nadd_tot = np.sum(nadd)
        ind_rest = np.cumsum(nadd)[rest > 0] - 1

        r_nadd = np.repeat(r, nadd)
        sep_nadd = np.repeat(sep, nadd)
        percon_nadd = np.repeat(percon, nadd)
        percon_nadd[ind_rest] = rest[rest > 0]



        # self.log(1, f"percon total = {2 * np.sum(percon_nadd)}")

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
        # noise = rng.normal(scale=scale_nadd).astype(np.int64)
        # noise = np.zeros_like(percon_nadd)

        # add both polarities of spots
        aflux = np.r_[percon_nadd + noise, -percon_nadd - noise]

        # IDL code has remainder concentration w/o noise...
        aflux[ind_rest] = aflux[ind_rest] - noise[ind_rest]
        aflux[ind_rest + nadd_tot] = aflux[ind_rest + nadd_tot] + noise[
            ind_rest]
        aflux = np.astype(aflux, np.int64)

        self.log(1, f"\tdelta nflux: {len(aflux):+6d} / {len(flux):6d}\tdelta flux: {np.sum(np.abs(aflux)):+7d} / {np.sum(np.abs(flux)):7d}")

        return aphi, atheta, aflux
