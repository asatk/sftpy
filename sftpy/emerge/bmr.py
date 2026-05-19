"""

"""

import abc
import numpy as np

from matplotlib import pyplot as plt
from ..viz import plot_syn

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..cycle import Cycle
from ..util import powerlaw_rv, schrijver_rv
from ._nesting import identify_nesting_plages

as_specified = rc["schrijver.as_specified"]

binflux= rc["physics.binflux"]
nfluxmax = rc["general.nfluxmax"]
dt = rc["general.dt"]
loglvl = rc["general.loglvl"]

cyl_mult = rc["cycle.mult"]


# orientation
joy = rc["schrijver.joy"]
joy_width = rc["schrijver.joy_width"]
joy_fold = rc["schrijver.joy_fold"]
sjzero = rc["schrijver.sjzero"]
max_lat = rc["schrijver.max_lat"]
lat_width = rc["schrijver.lat_width"]
lat_fold = rc["schrijver.lat_fold"]
turbulent = rc["schrijver.turbulent"]
psource = rc["schrijver.psource"]
avefluxd = rc["schrijver.avefluxd"]
miniflux = rc["schrijver.miniflux"]
maxflux = rc["schrijver.maxflux"]

thr = rc["rwalk.thr"]

rad = rc["physics.rad"]

thetabins = rc["synoptic.thetabins"]
phibins = rc["synoptic.phibins"]



class BMREmerge(Component, metaclass=abc.ABCMeta):
    """
    Base class for Bipole Magnetic Region emergence components that follow
    Schrijver's recipes and patterns of flux emergence.
    """

    prefix = "[bmr]"

    def __init__(self,
                 dt: float=dt,
                 nfluxmax: int=nfluxmax,
                 loglvl: int=loglvl):
        super().__init__(loglvl)
        self._dt = dt
        self._nfluxmax = nfluxmax

    @abc.abstractmethod
    def emerge(self,
               phi: np.ndarray,
               theta: np.ndarray,
               flux: np.ndarray,
               nflux: int):
        ...


class BMRAssimilate(BMREmerge):
    """
    Component for BMR emergence that assimilates solar magnetogram data.
    """

    prefix = "[bmr-assim]"

    def __init__(self,
                 dt: float=dt,
                 nfluxmax: int=nfluxmax,
                 loglvl: int=loglvl):
        super().__init__(dt, nfluxmax, loglvl)

    def emerge(self,
               phi: np.ndarray,
               theta: np.ndarray,
               flux: np.ndarray,
               nflux: int):
        ...



# TODO specified sources injection (pre step 1)
# if sources are specified, set variables and inject sources
"""
    if specified is not None:
        newflux = np.round(specified[:,0]).astype(np.int64)
        newphi = specified[:,1]
        newtheta = specified[:,2]
        ntotal = specified.shape[0]
        orient = np.full(ntotal, joy * np.pi / 180, dtype=np.float64)
        hemi_south = newtheta > np.pi / 2
        orient[hemi_south] = np.pi - orient[hemi_south]
        # assume the orientation of the largest cycle for the new regions
        orient += np.pi * np.all(source < 0)

        # inject sources
"""

# TODO pre step 3
"""
# assimilating
# remove sources from within radassim deg of the magnetrograph
# subobservation point
if assimilation:
    l0 = 0.0
    b0 = 0.0

    # TODO find default value for this... not in SFT Documentation
    # nor in code... just an update comment in addsources.pro
    radassim = 60.0

    xe, ye, ze = phithetaxyz(newphi + l0, newtheta, ntotal)
    pos = tiltmatrix(b0) @ np.array([[xe], [ye], [ze]])
    # shouldn't this be squared? sq. deg?
    edge = np.sin(radassim * np.pi / 180)
    ind = ((pos[:, 1] ** 2 + pos[:, 2] ** 2) < edge) & (pos[:, 0] > 0)
    # set source fluxes to zero if within assimilated region, then remove
    newflux[ind] = 0

    newflux = newflux[~ind]
    newphi = newphi[~ind]
    newtheta = newtheta[~ind]
    ntotal = len(newflux)
"""


class BMRSchrijver(BMREmerge):
    """
    Component for BMR emergence according to CJS empirical recipes
    """

    prefix = "[bmr-cjs]"

    def __init__(self,
                 cycle: Cycle,
                 dt: float=dt,
                 nfluxmax: int=nfluxmax,
                 as_specified: bool=False,
                 gradual: bool=False,
                 loglvl: int=loglvl):
        super().__init__(dt, nfluxmax, loglvl)
        self._cycle = cycle
        self._as_specified = as_specified   # fast forward/no ER/not full res
        self._gradual = gradual

    @property
    def as_specified(self):
        return self._as_specified

    @as_specified.setter
    def as_specified(self, value):
        self._as_specified = value

    def emerge(self,
               phi: np.ndarray,
               theta: np.ndarray,
               flux: np.ndarray,
               nflux: int):

        dt = self._dt
        gradual = self._gradual

        nflux_pre = nflux
        flux_pre = np.sum(np.abs(flux[:nflux]))

        source, latsource = self._cycle.cycle()

        # TODO what can we vectorize / pull out of loop?
        for i in range(len(source)):

            if np.abs(source[i]) < 1e-5:
                continue

            # Step 1 --- determine size distribution

            ## [1] High-flux tail dominant for large regions
            a = 8.0 * np.abs(source[i])
            p = psource
            pm1 = p - 1


            minflux = miniflux / binflux
            scale = (1.5 * avefluxd) ** pm1 / -pm1
            rangefactor = maxflux ** -pm1 - miniflux ** -pm1
            ntotal1 = 2 * a * dt / 86400 * scale * rangefactor

            frac = ntotal1 - int(ntotal1)
            ntotal1 = int(ntotal1) + (rng.uniform() < frac)

            # rv1 = powerlaw_rv(ntotal1, -p, minflux / 2 / binflux, maxflux / 2 / binflux, rng)
            rv1 = schrijver_rv(ntotal1, p, minflux / 2 / binflux, maxflux / 2 / binflux, rng)
            newflux1 = np.astype(rv1, np.int64)


            ## [2] Low-flux tail dominated by ephemeral regions
            a = 8.0
            a *= np.abs(source[i])**(1/3) * turbulent + (1 - turbulent)
            p = psource + 1
            pm1 = p - 1
            scale = (1.5 * avefluxd) ** pm1 / -pm1
            rangefactor = maxflux ** -pm1 - miniflux ** -pm1
            ntotal2 = 2 * a * dt / 86400 * scale * rangefactor

            frac = ntotal2 - int(ntotal2)
            ntotal2 = int(ntotal2) + (rng.uniform() < frac)

            # rv2 = powerlaw_rv(ntotal2, -p, minflux / 2 / binflux, maxflux / 2 / binflux, rng)
            rv2 = schrijver_rv(ntotal2, p, minflux / 2 / binflux, maxflux / 2 / binflux, rng)
            newflux2 = np.astype(rv2, np.int64)

            newflux = np.r_[newflux1, newflux2]
            ntotal = len(newflux)

            # accelerated time mode -- include only regions larger than 2sq deg
            # or 2 * 1.5e18 & avefluxd = 3 avefluxd units of 10^18 Mx/m^2
            # IDL model behavior includes all and only ephemeral regions if
            # cycle source strength relative to Sun is negative

            self.log(1, f"Cycle ({i}) strength: {source[i]:.05f}")
            self.log(1,
                     f"Active = {ntotal1}\t" + \
                     f"Ephemeral = {ntotal2}\t" + \
                     "All = {ntotal}")

            # fast forward stuff from old model
            # only emerge active regions
            if not self._as_specified and cyl_mult > 0:
                ind_big = np.nonzero(newflux > (3 * avefluxd / binflux))[0]
                if len(ind_big) == 0:
                    return phi, theta, flux, nflux
                newflux = newflux[ind_big]
                ntotal = len(newflux)

            # old test mode for negative cycle mult which should be a flag; alas
            # only emerge ephemeral regions
            if cyl_mult < 0:
                ind_small = np.nonzero(newflux < (3 * avefluxd / binflux))[0]
                if len(ind_small) == 0:
                    return phi, theta, flux, nflux
                newflux = newflux[ind_small]
                ntotal = len(newflux)


            if ntotal == 0:
                continue

            # Step 2 --- determine positions
            newphi = rng.uniform(high=2*np.pi, size=ntotal)
            newtheta = latsource[i] * np.pi / 180 * rng.choice([-1, 1], size=ntotal)
            width = lat_width * (np.exp(-newflux * binflux / lat_fold) + 0.15)
            newtheta += rng.normal(scale=width*np.pi/180, size=ntotal)
            # TODO introduced this myself just to prevent stuff from going oob
            # newtheta = np.clip(newtheta, a_min=-np.pi/2, a_max=np.pi/2)
            # latitude -> co-latitude
            newtheta = (np.pi/2 - newtheta) % np.pi

            # nesting
            # ~40% of activate regions emerge inside existing regions.
            # applied to all regions larger than 2.5 sq deg (factor 2 for 2 pol)
            # 1.4752 is flux to G
            active_thr = 2.5 * avefluxd * 1.47562 / 2 / binflux
            is_active = np.nonzero(newflux >= active_thr)[0]
            nactive = len(is_active)
            if nactive > 0:
                # pick nest regions from set of sufficiently large regions
                will_nest = rng.uniform(size=nactive) < 0.4
                nnest = np.sum(will_nest)

                self.log(3, f"NEST nactive = {nactive}")

                # pick new location inside plage regions but not at polar caps
                # limits emergence to lat +/- deg
                if nnest > 0:
                    is_nesting = is_active[will_nest]
                    self.log(3, f"NEST nnest = {nnest}")

                    # NOTE: nesting plages identified in IDL model much earlier
                    # than immediately after sampling new spots. before flows,
                    # fragmentation, and collisions.

                    nest_lat_lim = 50.0
                    is_plage = identify_nesting_plages(
                        phi, theta, flux, nflux, thr, binflux,
                        phibins, thetabins, nest_lat_lim)
                    is_plage_px = np.nonzero(np.ravel(is_plage))[0]
                    nplage = len(is_plage_px)

                    if self._loglvl >= 3:
                        plot_syn(phi, theta, flux, nflux, show=True)

                    self.plot(3, "imshow", is_plage.T)
                    self.pshow(3)

                    self.log(3, f"NEST nplage = {nplage}")

                    if nplage > 0:
                        nreplace = min(nnest, nplage)
                        self.log(3, f"NEST nreplace = {nreplace}")
                        point = rng.choice(is_plage_px, replace=False, size=nreplace)
                        point = np.astype(point, np.int64)
                        lat = point // phibins

                        # TODO check the  +1 on these
                        nest_newphi = point - phibins * lat
                        nest_newtheta = np.pi / 2 - np.arcsin(lat / (thetabins / 2) - 1)
                        newphi[is_nesting[:nreplace]] = nest_newphi
                        newtheta[is_nesting[:nreplace]] = nest_newtheta

                        self.log(3, f"NEST point: {point}")
                        self.log(3, f"NEST lat: {lat}")
                        self.log(3, f"NEST newphi: {nest_newphi}")
                        self.log(3, f"NEST newlat: {nest_newtheta}")

                    else:
                        self.log(3, "NEST no plage regions")
                else:
                    self.log(3, "NEST no nesting regions")
            else:
                self.log(3, "NEST no new active regions")

            # Step 3 --- orientation of bipole axes
            width = joy_width * np.exp(-binflux * newflux / joy_fold) + sjzero
            orient = rng.normal(loc=joy, scale=width, size=ntotal) * np.pi / 180
            # flip sign for opposite polarity regions in different hemispheres (Hale's Law)
            hemi = np.sign(np.pi / 2 - newtheta)
            orient = np.pi * (1 - hemi) / 2 + hemi * orient
            # invert polarity based on phase of cycle
            orient += np.pi * (source[i] < 0)



            # Step 4 --- position concentrations
            r = (np.sqrt(newflux * binflux * 1e18 / avefluxd / np.pi) + 7e8) / 7e10
            # impose minimum separation of ~0.5 supergranulation of 18Mm
            sep = np.clip(r, a_min=9000/rad/2, a_max=None)
            # number of new concentrations that contain 15e18 Mx w/ at least
            # three equal concentrations per polarity
            percon = np.clip(newflux / 3., a_min=1, a_max=None)
            percon[percon > (15. / binflux)] = 15. / binflux

            # bulk = np.clip(newflux // percon, a_min=1, a_max=None)
            bulk = np.clip(np.astype(
                newflux / percon, np.int64),
                a_min=1, a_max=None)
            rest = np.clip(newflux - percon * bulk, a_min=0, a_max=None)

            nadd = bulk + (rest > 0)
            nadd[newflux < bulk * percon] = 1
            ind_rest = np.cumsum(nadd) - 1

            r_nadd = np.repeat(r, nadd)
            sep_nadd = np.repeat(sep, nadd)
            percon_nadd = np.repeat(percon, nadd)
            percon_nadd[ind_rest] = rest
            nadd_tot = ind_rest[-1] + 1

            # one polarity
            offset1 = rng.uniform(high=r_nadd)
            angle1 = rng.uniform(high=2*np.pi, size=nadd_tot)

            # opposite polarity
            offset2 = rng.uniform(high=r_nadd)
            angle2 = rng.uniform(high=2*np.pi, size=nadd_tot)

            x_tmp = np.r_[ sep_nadd + offset1 * np.cos(angle1),
                          -sep_nadd + offset2 * np.cos(angle2)]
            y_tmp = np.r_[offset1 * np.sin(angle1),
                          offset2 * np.sin(angle2)]

            # orientation of bipolar spot
            #TODO better way to double these?
            orient_nadd_half = np.repeat(orient, nadd)
            orient_nadd = np.r_[orient_nadd_half, orient_nadd_half]

            orient_tmp = orient_nadd + np.pi / 2
            coso = np.cos(orient_tmp)
            sino = np.sin(orient_tmp)
            xo = coso * x_tmp + sino * y_tmp
            yo = -sino * x_tmp + coso * y_tmp

            # location of bipolar active region / concentration
            newphi_nadd_half = np.repeat(newphi, nadd)
            newphi_nadd = np.r_[newphi_nadd_half, newphi_nadd_half]
            newtheta_nadd_half = np.repeat(newtheta, nadd)
            newtheta_nadd = np.r_[newtheta_nadd_half, newtheta_nadd_half]

            cosphi = np.cos(newphi_nadd)
            sinphi = np.sin(newphi_nadd)
            costheta = np.cos(newtheta_nadd)
            sintheta = np.sin(newtheta_nadd)

            # Cartesian coordinates of spots
            x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
            y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
            z = costheta - xo * sintheta

            # spherical coordinates of spots
            aphi = np.arctan2(y, x) % (2 * np.pi)
            atheta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

            # Poisson noise added to each concentration
            scale_nadd = np.sqrt(percon_nadd)
            noise = rng.normal(scale=scale_nadd)

            # add both polarities of spots
            aflux = np.r_[percon_nadd + noise, -percon_nadd - noise]

            # IDL code has remainder concentration w/o noise...
            aflux[ind_rest] = aflux[ind_rest] - noise[ind_rest]
            aflux[ind_rest + nadd_tot] = aflux[ind_rest + nadd_tot] + noise[ind_rest]
            aflux = np.astype(aflux, np.int64)

            # self.log(1, f"flux sum: {np.sum(np.abs(aflux)):.4e}")
            
            # TODO vectorize this loop for gradual
            # gradual introduction of active regions
            if gradual:
                for j in range(ntotal):

                    # TODO add gradual emergence param
                    dur = np.asarray(np.sum(np.abs(aflux)) * binflux / 0.05 / dt, dtype=np.int64)+1
                    
                    if dur > 1 or np.any((latime == 0) & (laflux != 0)):
                        
                        self.log(2, f"gradual flux emergence")

                        an = len(aflux) / 2
                        ai = rng.integers(high=dur, size=an)
                        laphi = np.r_[laphi, aphi]
                        latheta = np.r_[latheta, atheta]
                        laflux = np.r_[laflux, aflux]
                        latime = np.r_[latime, ai, ai]

                        ia = (latime == 0) & (laflux != 0)
                        if np.any(ia):
                            aphi = laphi[ia]
                            atheta = latheta[ia]
                            aflux = laflux[ia]
                            nadd = np.sum(ia) / 2
                        else:
                            aphi = 0.0
                            atheta = 0.0
                            aflux = 0
                            nadd = 0

                        ia = (latime != 0) & (laflux != 0)
                        if np.any(ia):
                            laphi = laphi[ia]
                            latheta = latheta[ia]
                            laflux = laflux[ia]
                            latime = latime[ia]
                        else:
                            laphi = 0.0
                            latheta = 0.0
                            laflux = 0
                            latime = 0

            if nflux + 2 * nadd_tot >= self._nfluxmax:
                
                while nflux + 2 * nadd_tot >= self._nfluxmax:
                    self._nfluxmax *= 2

                self.log(2, f"nfluxmax {self._nfluxmax}")

                phi_cp = np.empty(self._nfluxmax)
                theta_cp = np.empty(self._nfluxmax)
                flux_cp = np.empty(self._nfluxmax)

                phi_cp[:nflux] = np.copy(phi[:nflux])
                theta_cp[:nflux] = np.copy(theta[:nflux])
                flux_cp[:nflux] = np.copy(flux[:nflux])

                phi = phi_cp
                theta = theta_cp
                flux = flux_cp

            # self.log(0, f"added nspots: {nadd_tot}")

            phi[nflux:nflux+2*nadd_tot] = aphi
            theta[nflux:nflux+2*nadd_tot] = atheta
            flux[nflux:nflux+2*nadd_tot] = aflux

            nflux += nadd_tot * 2

            # self.log(1, f"add {nadd_tot}")

        nflux_post = nflux
        flux_post = np.sum(np.abs(flux[:nflux]))

        self.log(1, f"delta nflux: {nflux_post - nflux_pre:6d} / {nflux_pre}\t" + \
                 f"delta flux : {flux_post - flux_pre:7d} / {flux_pre}")


        return phi, theta, flux, nflux
