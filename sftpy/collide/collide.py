"""
Collision schemes from the original Schrijver+ model:
 - COL1: sequentially coalesce nearby spots into randomly-selected final spot.
 - COL2: ?
 - COL3: like COL2 but in C

"""

# TODO
# parameter in kit_iocontrol.pro `collide`
# Collisions 0-none 1-equal polatiry, 2-both polarities

import abc
import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component

dt = rc["general.dt"]
correction = rc["collide.correction"]
meanv = rc["collide.meanv"]
diffusion = rc["physics.diffusion"]
loglvl = rc["component.loglvl"]


class Collide(Component, metaclass=abc.ABCMeta):
    """
    Base class for flux concentration collision component of computation sequence.
    """

    prefix = "[collide]"

    def __init__(self,
                 dt: float=dt,
                 correction: float=correction,
                 meanv: float=meanv,
                 diffusion: float=diffusion,    # TODO check if diff is in IDL code
                 trackcancel: bool=False,
                 loglvl: int=loglvl):

        super().__init__(loglvl)
        self._dt = dt
        self._corr = correction
        # TODO same constant as in charges/charges
        self._difu = diffusion
        self._track = trackcancel

        # collision param from schrijver+ 1997 l=1400km^2/s
        self._radius = 1400 / meanv * correction

        self._crphi = self._radius / 7e5
        self._critical = self._crphi ** 2 # collision distance in units of stellar radius squared

    def _collide_start(self,
                       phi: np.ndarray,
                       theta: np.ndarray,
                       flux: np.ndarray,
                       nflux: int):

        # coordinates
        sintheta = np.sin(theta[:nflux])
        x = sintheta * np.cos(phi[:nflux])
        y = sintheta * np.sin(phi[:nflux])
        z = np.cos(theta[:nflux])
        r = np.stack([x, y, z], axis=1)

        return r

    def _collide_finish(self,
                        phi: np.ndarray,
                        theta: np.ndarray,
                        flux: np.ndarray,
                        nflux: int):

        nfluxold = nflux
        index = np.nonzero(flux[:nflux])[0]
        nnew = len(index)
        if nflux == nnew:
            return nflux

        self.log(2, f"[collide finish] ---- nnew {nnew}/{nflux}")

        phi[:nnew] = phi[index]
        theta[:nnew] = theta[index]
        flux[:nnew] = flux[index]

        return nnew

    @abc.abstractmethod
    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):
        ...


class COLNone(Collide):

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        r = self._collide_start(phi, theta, flux, nflux)
        return self._collide_finish(phi, theta, flux, nflux)

class COLScan(Collide):
    """
    Collide by scanning over phi
    """

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):


        # permutation of indices based on value of phi
        perm = np.argsort(phi[:nflux])
        psort = phi[perm]
        tsort = theta[perm]
        fsort = flux[perm]

        r = self._collide_start(psort, tsort, fsort, nflux)

        # tunable param: size of 2D arrays for computation. choose btwn 1k-10k
        n = 1000


        minphi = 0.0
        maxphi = phi[:n]

        lo = 0
        hi = n

        while hi < nflux:

            # spots of opposite-signed flux that can "collide"
            signs = np.add.outer(np.sign(fsort[lo:hi]), np.sign(fsort[lo:hi]))
            ind_opp = signs == 0

            # determine distances between spots
            dists = np.sqrt(np.sum(np.square(np.apply_along_axis(
                lambda arr: np.subtract.outer(arr, arr), 0, r)), axis=2))
            ind_close = dists < self._critical

            # exclude connection to self
            ind_self = ~(np.eye(hi-lo) == 1)

            # spots that collide have opposite-polarity fluxes and are nearby
            ind_col = ind_opp & ind_close & ind_self

            # locate concentrations that can collide (have opposite-pol neighbors)
            where_col = np.nonzero(np.any(ind_col, axis=1))[0]

            # coalesce spots until none remain
            while len(where_col) > 0:

                # choose one spot to be the "hub" into which its neighbors coalesce
                hub = rng.choice(where_col)

                # identify neighbors
                nbrs = np.nonzero(ind_col[hub])[0]

                # calculate total flux to be added to hub spot
                hubflux = np.sum(fsort[nbrs])
                
                # add coalesced flux to hub
                fsort[hub] += hubflux

                # remove connections to coalesced spots
                ind_col[nbrs] = False
                ind_col[:,nbrs] = False     # could potentially remove w/ slick coding

                # zero out coalesced spots
                fsort[nbrs] = 0

                # locate concentrations that can collide after coalescing prev
                where_col = np.nonzero(np.any(ind_col, axis=1))[0]
            ...


class COL1(Collide):
    """

    """

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        r = self._collide_start(phi, theta, flux, nflux)

        # determine pos/neg concentrations
        indp = np.nonzero(np.sign(flux[:nflux]) == +1)[0]
        indn = np.nonzero(np.sign(flux[:nflux]) == -1)[0]

        self.log(1, f"npos {len(indp)} / {nflux}   nneg {len(indn)} / {nflux}")

        rn = r[indn]
        fluxn = flux[indn]

        # iterate through pos concentrations
        for i in indp:

            # determine neg neighbors to pos spot i
            nbrs = np.sum(np.square(r[i] - rn), axis=1) < self._critical

            # shuffle list of spots to randomly determine one spot for all to
            # coalesce into and the others for removal
            spots = rng.permutation(np.r_[indn[nbrs], i])

            # calculate total flux in collided spots
            sumflux = np.sum(flux[spots])

            # hub spot has combined flux
            hubspot = spots[0]

            # remaining spots to be deleted
            restspots = spots[1:]

            flux[hubspot] = sumflux
            flux[restspots] = 0


        """

        # TODO vectorize
        for i in range(nflux-1):
            
            self.log(1, f"spot {i}")

            # skip empty flux concentration
            if flux[i] == 0:
                continue

            ind1 = np.sign(flux[i]) + np.sign(flux[i+1:nflux]) == 0
            ind1 = np.nonzero(ind1)[0] + i + 1

            ind2 = np.sum(np.abs(r[:,i,None] - r[:,ind1]), axis=0) < self._cr
            ind2 = np.nonzero(ind2)[0] + i + 1

            '''
            ind1 = np.nonzero(
                    (np.sign(flux[i]) != np.sign(flux[i+1:nflux])) & \
                    (flux[i+1:nflux] != 0))[0] + i+1

            # TODO switch order of dims for r/rb
            ind2 = np.nonzero(
                    (np.sum(np.abs(r[:,i,None] - r[:,ind1]), axis=0) < self._cr) & \
                    (flux[i+1:nflux] != 0))[0] + i+1
            '''

            if np.any(ind2):
                ind3 = np.nonzero(
                        np.sum(np.square(r[:,i,None] - r[:,ind2]), axis=0) < self._critical)[0]

                if np.any(ind3):
                    flux[i] += np.sum(flux[ind2[ind3]])
                    flux[ind2[ind3]] = 0    # eliminate collided particle(s)

                    # TODO there are some -1s in the IDL... compare code snippets
                    ic = rng.choice(ind3)
                    # TODO check this is just a not empty condition
                    if ic > 0:
                        phi[i] = phi[ind2[ic]]
                        theta[i] = theta[ind2[ic]]

        """

        return self._collide_finish(phi, theta, flux, nflux)


class COL2(Collide):

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        r = self._collide_start(phi, theta, flux, nflux)

        for i in range(nflux-1):

            # skip empty flux concentration
            if flux[i] == 0:
                continue

            thetalo = theta[i] - self._crphi
            lo = max(i-25, 0)

            # do some mod arithmetic
            while (theta[lo] > thetalo and lo > 0):
                lo = max(lo-25, 0)

            thetahi = theta[i] + self._crphi
            hi = min(i+25, nflux-1)

            # do some mod arithmetic
            while (theta[hi] < thetahi and hi < nflux-1):
                hi = min(hi+25, nflux-1)

            lcrphi = self._crphi / np.sin(theta[i])

            ind1 = np.nonzero(np.fabs(phi[i] - phi[lo:hi+1]) < lcrphi)[0] + lo
            ind2 = np.nonzero(
                    (np.sum(np.square(r[:,[i]] - r[:, ind1])) < critical) & \
                    (flux[ind1] != 0))[0]

            if self._track:
                cancelflux = np.fabs(np.sum(np.fabs(flux[ind2])) - np.sum(flux[ind2]))/2
                # TODO check if this is just a non-empty condition
                # TODO what is this
                if cancelflux > 0:
                    self.log(1, f"[collide] ---- cancelflux {cancelflux}")

            flux[i] = np.sum(flux[ind2])
            n = len(ind2)

            # eliminate coalesced concentrations
            if n > 1:
                flux[ind2[ind2 != i]] = 0

                ic = rng.choice(ind2)
                phi[i] = phi[ic]
                theta[i] = theta[ic]

        return self._collide_finish(phi, theta, flux, nflux)


class COL3(Collide):

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        r = self._collide_start(phi, theta, flux, nflux)

        # TODO compiled C code -- just write in py for now
        # TODO um
        nflux = long(nflux)
        # TODO but this isnt nflux?
        #order = np.zeros(len(flux))

        order = rng.permutation(np.arange(nflux, dtype=np.int64))

        for i in range(nflux-1):
            
            ind = order[i]
            
            if flux[ind] == 0:
                continue

            thetalo = theta[ind] - self._crphi
            lo = ind-25
            if lo < 0:
                lo = 0

            while (theta[lo] > thetalo and low > 0):
                lo = lo - 25
                if lo < 0:
                    lo = 0

            thetahi = theta[ind] + self._crphi
            hi = ind+25
            if hi > nflux-1:
                hi = nflux-1

            while (theta[hi] < thetahi and hi < nflux-1):
                hi = hi+25
                if hi > nflux - 1:
                    hi = nflux-1

            lcrphi = self._crphi / np.sin(theta[ind])

            k = 0
            for j in range(lo, hi):
                d = phi[ind] - phi[j]
                if d < 0:
                    d = -d
                if d < lcrphi:
                    # TODO what is index?
                    index[k] = j
                    k += 1

            n = 0
            for j in range(k):
                ind2 = index[j]
                d = np.sum(np.square(r[:, ind] - r[:, ind2]))
                if d < self._critical and flux[ind] != 0:
                    index[n] = ind2
                    n += 1

            # TODO rng.choice? what is index (wht is length)
            ic = rng.integers(n)
            ic = index[ic]

            for j in range(n):
                if ic != index[j]:
                    ind2 = index[j]
                    flux[ic] += flux[ind2]
                    flux[ind2] = 0

        return self._collide_finish(phi, theta, flux, nflux)
