"""

"""

import abc
import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..cycle import Cycle
from .regions import MagneticRegion
from ._nesting import PlageNests


binflux= rc["physics.binflux"]
nfluxmax = rc["general.nfluxmax"]
dt = rc["general.dt"]
loglvl = rc["general.loglvl"]

cyl_mult = rc["cycle.mult"]









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
                 region: MagneticRegion,
                 nest: PlageNests,
                 dt: float=dt,
                 nfluxmax: int=nfluxmax,
                 gradual: bool=False,
                 loglvl: int=loglvl):
        super().__init__(dt, nfluxmax, loglvl)
        self._cycle = cycle
        self._gradual = gradual
        self._region = region
        self._nest = nest


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

            # Step 1 --- determine bipole size distribution
            self.log(1, f"Cycle ({i}) strength: {source[i]:.05f}")
            newflux = self._region.sample_flux(source[i])
            ntotal = len(newflux)

            if ntotal == 0:
                continue

            # Step 2 --- determine positions
            newphi = self._region.sample_phi(None, ntotal)
            newtheta = self._region.sample_theta(newflux, latsource[i], ntotal)

            # nesting
            # ~40% of activate regions emerge inside existing regions.
            # applied to all regions larger than 2.5 sq deg
            self._nest.place_active_regions(
                phi, theta, flux, nflux, newphi, newtheta, newflux)


            # Step 3 --- orientation of bipole axes
            orient = self._region.sample_orientation(
                newphi, newtheta, newflux, source[i], ntotal)

            self.log(1, f"newflux = {np.sum(2 * newflux)}")

            # Step 4 --- position concentrations
            aphi, atheta, aflux = self._region.make_concentrations(
                newphi, newtheta, newflux, orient)
            nadd_tot = len(aphi) // 2

            # self.log(1, f"flux sum: {np.sum(np.abs(aflux)):.4e}")

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


            self.log(1, f"aflux = {np.sum(np.abs(aflux))}")

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
