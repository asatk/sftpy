"""

"""

import abc
import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..util import powerlaw_rv
from ..util.other import tiltmatrix, phithetaxyz

binflux= rc["physics.binflux"]
nfluxmax = rc["general.nfluxmax"]
# TODO change - just checking if cycle exists and what dir/sign?
cyl_mult = rc["cycle.mult"]
dt = rc["general.dt"]
loglvl = rc["general.loglvl"]

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
               nflux: int,
               source: np.ndarray,
               latsource: np.ndarray,
               synoptic: np.ndarray,):
        ...



class BMRNone(BMREmerge):

    def emerge(self,
               phi: np.ndarray,
               theta: np.ndarray,
               flux: np.ndarray,
               nflux: int,
               source: np.ndarray,
               latsource: np.ndarray,
               synoptic: np.ndarray):
        return phi, theta, flux, nflux



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
               nflux: int,
               source: np.ndarray=None,
               latsource: np.ndarray=None,
               synoptic: np.ndarray=None):
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
                 dt: float=dt,
                 nfluxmax: int=nfluxmax,
                 as_specified: bool=True,
                 gradual: bool=False,
                 loglvl: int=loglvl):
        super().__init__(dt, nfluxmax, loglvl)
        self._as_specified = as_specified   # fast forward/ no ER/ not full res
        self._gradual = gradual

    def emerge(self,
               phi: np.ndarray,
               theta: np.ndarray,
               flux: np.ndarray,
               nflux: int,
               source: np.ndarray,
               latsource: np.ndarray,
               synoptic: np.ndarray):

        dt = self._dt
        as_specified = self._as_specified
        gradual = self._gradual

        # track total flux added in timestep
        # sourceinput = 0.0

        # TODO what can we vectorize / pull out of loop?
        for i in range(len(source)):

            # cycle strength is negligible; inject no new sources
            # if np.abs(source[i]) < 1e-5:
            #     continue



            # Step 1 --- determine size distribution

            ## [1] High-flux tail dominant for large regions
            a = 8.0
            a *= np.abs(source[i])
            p = psource


            minflux = miniflux / binflux
            scale = 1. / ((1.0 - p) * 1.5 ** (1.0 - p) * avefluxd ** (1.0 - p))
            rangefactor = maxflux ** (1.0 - p) - (miniflux * 2) ** (1.0 - p)
            ntotal1 = 2 * a * dt / 86400 * scale * rangefactor

            frac = ntotal1 - int(ntotal1)
            ntotal1 = int(ntotal1) + (rng.uniform() < frac)

            rv1 = powerlaw_rv(ntotal1, -p, minflux / binflux, maxflux / binflux, rng)
            newflux1 = np.astype(rv1, np.int64)


            ## [2] Low-flux tail dominated by ephemeral regions
            a = 8.0
            a *= np.abs(source[i])**(1.0/3) * turbulent + (1 - turbulent)
            p = psource + 1
            scale = 1. / ((1.0 - p) * 1.5 ** (1.0 - p) * avefluxd ** (1.0 - p))
            rangefactor = maxflux ** (1.0 - p) - (miniflux * 2) ** (1.0 - p)
            ntotal2 = 2 * a * dt / 86400 * scale * rangefactor

            frac = ntotal2 - int(ntotal2)
            ntotal2 = int(ntotal2) + (rng.uniform() < frac)

            rv2 = powerlaw_rv(ntotal2, -p, minflux / binflux, maxflux / binflux, rng)
            newflux2 = np.astype(rv2, np.int64)

            newflux = np.r_[newflux1, newflux2]
            ntotal = len(newflux)
        
            if ntotal == 0:
                continue

            self.log(1, f"ntotal1 = {ntotal1}   ntotal2 = {ntotal2}   ntotal = {ntotal}")
            self.log(1, f"ntotal (all) = {ntotal}")
            self.log(3, f"flux stats: mean {np.mean(newflux)}" + \
                    f"   median {np.median(newflux)}   sd {np.std(newflux)}")
            self.plot(3, "hist", newflux, bins=50)
            self.plot(3, "xlim", (0.0, None))
            self.plot(3, "title", "newflux")
            self.plot(3, "show")

            # TODO -- this mode emerges nothing...
            # accelerated-time mode; leave out ephemeral regions
            # regions that are 2 sq deg or larger, i.e.,
            # 2*1.5e18*avefluxd/1e18 = 3 avefluxd units of 1e18
            if not as_specified and cyl_mult > 1e-5:
                self.log(2, f"NEWFLUX {newflux}")
                index = newflux > (3 * avefluxd / binflux)
                if not np.any(index):
                    self.log(2, "NO SOURCES PASS THRESHOLD")
                    return phi, theta, flux, nflux
                newflux = newflux[index]
                ntotal = len(newflux)

            # testrun mode -- include only ephemeral regions
            if cyl_mult < -1e-5:
                index = newflux < (3 * avefluxd / binflux)
                if not np.any(index):
                    return phi, theta, flux, nflux
                newflux = newflux[index]
                ntotal = len(newflux)



            # Step 2 --- determine positions
            self.log(1, f"latsource = {latsource[i]}")
            newphi = rng.uniform(high=2*np.pi, size=ntotal)
            newtheta = latsource[i] * np.pi / 180 * rng.choice([-1, 1], size=ntotal)
            width = lat_width * (np.exp(-newflux * binflux / lat_fold) + 0.15)
            newtheta += rng.normal(scale=width*np.pi/180, size=ntotal)
            # introdued this myself just to prevent stuff from going oob
            newtheta = np.clip(newtheta, a_min=-np.pi/2, a_max=np.pi/2)
            newtheta = np.pi/2 - newtheta

            self.log(3, "new theta")
            self.plot(3, "hist", newtheta, bins=25)
            self.plot(3, "title", "concentration latitudes")
            self.plot(3, "show")

            # nesting
            # ~40% of activate regions emerge inside existing regions.
            # applied to all regions larger than 2.5 sq deg (factor 2 for 2 pol)
            nesting = 2.5 * avefluxd * 1.47562 / 2 / binflux
            ind_nest = np.nonzero(newflux >= nesting)[0]
            nnest = len(ind_nest)
            if nnest > 0:
                # pick nest regions from set of sufficiently large regions
                ind_pick = rng.uniform(size=nnest) < 0.4
                npick = np.sum(ind_pick)

                # pick new location inside plage regions but not at polar caps
                # limits emergence to lat +/- deg
                if npick > 0:
                    ind_nest_picks = ind_nest[ind_pick]
                    self.log(2, f"nnest={nnest}   npick={npick}")
                    self.log(2, f"ind_nest {ind_nest}")
                    self.log(2, f"ind_pick {ind_pick}")
                    self.log(2, f"ind_nest_pick {ind_nest_picks}")

                    thetabins = 180
                    phibins = 360

                    thetalim = np.int64(np.sin(50 * np.pi / 180) * thetabins / 2) + 1

                    xx = np.zeros(thetabins, dtype=np.byte)
                    xx[thetalim:thetabins-thetalim] = 1
                    yy = np.ones(phibins, dtype=np.byte)
                    # TODO check this matmult
                    # ind = (synoptic * np.outer(yy, xx)) != 0
                    ind = np.nonzero(np.ravel((synoptic * np.outer(yy, xx))))[0]
                    nind = len(ind)

                    if nind > 0:
                        nreplace = min(npick, nind)
                        self.log(3, f"NEST nreplace = {nreplace}")
                        point = rng.choice(ind, replace=False, size=nreplace)
                        lat = point / phibins

                        # TODO check the  +1 on these
                        self.log(3, f"NEST point = {point.shape}")
                        self.log(3, f"NEST lat = {lat.shape}")
                        self.log(3, f"NEST ind_nest_picks = {ind_nest_picks.shape}")
                        self.log(3, f"NEST new_phi = {newphi.shape}")
                        newphi[ind_nest_picks[:nreplace+1]] = point - phibins * lat
                        newtheta[ind_nest_picks[:nreplace+1]] = np.pi / 2 - np.arcsin(lat / (thetabins / 2) - 1)
                    else:
                        self.log(3, "NEST no nests")
                else:
                    self.log(3, "NEST no nests")
                




            # Step 3 --- orientation of bipole axes
            width = joy_width * np.exp(-binflux * newflux / joy_fold) + sjzero
            orient = rng.normal(loc=joy, scale=width, size=ntotal) * np.pi / 180
            # flip sign for opposite polarity regions
            ind_pol = newtheta > (np.pi / 2)
            orient[ind_pol] = np.pi - orient[ind_pol]
            orient += np.pi * (source[i] < 0)

            self.log(1, f"joy = {joy}   joy_width = {joy_width}    joy_fold {joy_fold}")
            self.plot(3, "hist", orient, bins=15)
            self.plot(3, "title", "Orient")
            self.plot(3, "show")



            # Step 4 --- position concentrations
            r = (np.sqrt(binflux * newflux * 1.e18 / avefluxd / np.pi) + 7.e8) / 7.e10
            self.log(3, "separation")
            self.plot(3, "hist", r)
            self.plot(3, "title", "Separation r")
            self.plot(3, "show")
            # impose minimum separation of ~0.5 supergranulation of 18Mm
            sep = np.clip(r, a_min=9000./7.e5/2, a_max=None)
            # number of new concentrations that contain 15e18 Mx w/ at least
            # three equal concentrations per polarity
            percon = np.clip(newflux / 3, a_min=1, a_max=None).astype(np.int64)
            percon[newflux > 3 * 15 / binflux] = 15 / binflux

            bulk = np.clip(newflux // percon, a_min=1, a_max=None)
            rest = np.clip(newflux - percon * bulk, a_min=0, a_max=None)
            
            nadd = np.ones(ntotal, dtype=np.int64)
            nadd[newflux >= bulk * percon] = bulk + (rest > 0)

            r_nadd = np.repeat(r, nadd)
            sep_nadd = np.repeat(sep, nadd)
            percon_nadd = np.repeat(percon, nadd)
            nadd_tot = np.sum(nadd)

            # one polarity
            offset1 = rng.uniform(high=r_nadd)
            angle1 = rng.uniform(high=2*np.pi, size=nadd_tot)

            # opposite polarity
            offset2 = rng.uniform(high=r_nadd)
            angle2 = rng.uniform(high=2*np.pi, size=nadd_tot)

            x_tmp = np.r_[sep_nadd + offset1 * np.cos(angle1),
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

            x = cosphi * sintheta + xo * cosphi * costheta - yo * sinphi
            y = sinphi * sintheta + xo * sinphi * costheta + yo * cosphi
            z = costheta - xo * sintheta

            if self._loglvl >= 7:
                from matplotlib import pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                #oct1 = (x > 0.0) & (y > 0.0) & (z > 0.0)
                #oct1 = (x > 0.0) & (y > 0.0) & (z > 0.0)
                ax.scatter(x, y, z, s=5)
                #ax.scatter(x[oct1], y[oct1], z[oct1])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.show()

            aphi = np.fmod(np.arctan2(y, x) + 2 * np.pi, 2 * np.pi)
            atheta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

            self.log(5, f"aphi {aphi.shape}")
            self.log(5, f"atheta {atheta.shape}")
            self.log(5, f"mean atheta {np.mean(atheta)} std atheta "
                  f"{np.std(atheta)}")

            self.log(3, "atheta hist")
            self.plot(3, "hist", atheta, bins=150)
            self.plot(3, "xticks", [0, np.pi/2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
            self.plot(3, "xlim", (0, np.pi))
            self.plot(3, "title", "added spot latitudes")
            self.plot(3, "show")

            self.log(3, "aphi hist")
            self.plot(3, "hist", aphi, bins=150)
            self.plot(3, "xticks", [0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
            self.plot(3, "xlim", (0, 2*np.pi))
            self.plot(3, "title", "added spot longitudes")
            self.plot(3, "show")

            # TODO add Poisson noise CORRECTLY -- this caused the weird streak
            scale_nadd = np.sqrt(percon_nadd)
            noise = rng.normal(scale=scale_nadd)
            noise = 0.0

            # TODO uhh confirm this funky index magic i made up
            aflux = np.r_[percon_nadd + noise, -percon_nadd - noise]
            rest_nz = np.nonzero(rest)
            aflux[nadd[rest_nz]] = rest[rest_nz]
            aflux[(nadd+1+nadd_tot)[rest_nz]] = -rest[rest_nz]

            """
            if rest != 0:
                aflux[nadd] = rest
                aflux[nadd + 1 + nadd_tot] = -rest
            """
            
            # TODO vectorize this loop for gradual
            # gradual introduction of active regions
            if gradual:
                for j in range(ntotal):

                    dur = np.asarray(np.sum(np.fabs(aflux)) * binflux / 0.05 / dt)+1
                    
                    if dur > 1 or np.any((latime == 0) & (laflux != 0)):
                        
                        self.log(2, f"gradual flux emergence")

                        an = len(aflux) / 2
                        ai = np.asarray(rng.uniform(high=dur, size=an))
                        laphi = np.r_[laphi, aphi]
                        latheta = np.r_[latheta, atheta]
                        laflux = np.r_[laflux, aflux]
                        latime = np.r_[latime, ai, ai]

                        ia = (latime == 0) & (laflux != 0)
                        if np.any(ia):
                            aphi = laphi[ia]
                            atheta = latheta[ia]
                            aflux = laflux[ia]
                            nadd = len(ia) / 2
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


            self.log(1, f"nadd_tot {nadd_tot}")
            self.log(1, f"2*nadd_tot {2*nadd_tot}")
            self.log(1, f"nflux+2*nadd_tot {nflux+2*nadd_tot}")
            self.log(1, f"phi[nflux:nflux+2*nadd_tot] {phi[nflux:nflux+2*nadd_tot].shape}")
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

            phi[nflux:nflux+2*nadd_tot] = aphi
            theta[nflux:nflux+2*nadd_tot] = atheta
            flux[nflux:nflux+2*nadd_tot] = aflux

            nflux += nadd_tot * 2

            self.log(1, f"add {nadd_tot}")


        return phi, theta, flux, nflux
