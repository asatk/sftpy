import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component

dt = rc["general.dt"]
fragment = rc["fragment.fragment"]
correction = rc["fragment.correction"]
loglvl = rc["component.loglvl"]
binflux = rc["physics.binflux"]


class Fragment(Component):

    def __init__(self, rwalk, dt: float=dt, fragment: float=fragment,
                 correction: float=correction, loglvl: int=loglvl):
        super().__init__(loglvl=loglvl)
        self._dt = dt
        self._k0 = 0.4e-6 * binflux * fragment * correction

    def fragment(self, phi, theta, flux, nflux):

        aflux = np.abs(flux[:nflux])

        # old prob/lifetime had + 0.008 not 0.12
        prob = (aflux * np.exp(-aflux * binflux / 120.) + 0.12) * self._k0 * \
                self._dt

        # bernoulli trial for each spot's breakup
        frag = np.nonzero(rng.uniform(size=nflux) < prob)[0]
        nfrag = len(frag)

        if nfrag:
            return nflux

        # fraction of flux fragmented
        frac = np.astype(rng.uniform(high=0.5, size=nfrag)*flux[frag], np.int64)

        # don't fragment fractions of 0 or 1
        ind = (frac != 0) & (flux[frag] != frac)
        nnew = np.count_nonzero(ind)

        if nnew == 0:
            return nflux

        newfrag = frag[ind]
        newflux = frac[ind]

        # subtract new frag flux from existing frag
        flux[newfrag] -= newflux

        newphi = phi[newfrag].copy()
        newtheta = theta[newfrag].copy()
        
        # move new frags over a dist of `dd` km
        self._rwalk.move(newphi, newtheta, newfrag, nnew)

        # add new fragments to list of spots
        phi[nflux:nflux+nnew] = newphi
        theta[nflux:nflux+nnew] = newtheta
        flux[nflux:nflux+nnew] = frac2
        nflux += nnew

        return nflux
