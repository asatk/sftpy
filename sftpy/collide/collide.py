"""
Collision schemes from the original Schrijver+ model:
 - COL1: sequentially coalesce nearby spots into randomly-selected final spot.
 - COL2: ?
 - COL3: like COL2 but in C

"""

# TODO
# parameter in kit_iocontrol.pro `collide`
# Collisions 0-none 1-opposite polarity, 2-both polarities

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

    def __init__(self,
                 range: int=25,
                 dt: float = dt,
                 correction: float = correction,
                 meanv: float = meanv,
                 diffusion: float = diffusion,  # TODO check if diff is in IDL code
                 trackcancel: bool = False,
                 loglvl: int = loglvl):
        super().__init__(dt, correction, meanv, diffusion, trackcancel, loglvl)
        self._range = range

    @property
    def range(self):
        return self._range

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        # sorting is not the bottleneck
        sort_idx = np.argsort(theta[:nflux])
        phi[:nflux] = phi[sort_idx]
        theta[:nflux] = theta[sort_idx]
        flux[:nflux] = flux[sort_idx]

        r = self._collide_start(phi, theta, flux, nflux)
        order = rng.permutation(np.arange(nflux, dtype=np.int64))

        for i in order:

            if flux[i] == 0:
                continue

            self._log.clockstart(2)
            thetalo = theta[i] - self._crphi

            # ind_thetalo = np.nonzero(theta - thetalo > 0)[0]
            # if len(ind_thetalo) == 0:
            #     lo = 0
            # else:
            #     lo = ind_thetalo[0]

            lo = i - self._range
            if lo < 0:
                lo = 0

            while (theta[lo] > thetalo and lo > 0):
                lo = lo - self._range
                if lo < 0:
                    lo = 0

            # self._log.clockstop(2, "collide thetalo end")

            thetahi = theta[i] + self._crphi

            # ind_thetahi = np.nonzero(theta - thetahi > 0)[0]
            # if len(ind_thetahi) == 0:
            #     hi = nflux - 1
            # else:
            #     hi = ind_thetahi[0]

            hi = i + self._range
            if hi > nflux-1:
                hi = nflux-1

            while (theta[hi] < thetahi and hi < nflux-1):
                hi = hi + self._range
                if hi > nflux-1:
                    hi = nflux-1

            lcrphi = self._crphi / np.sin(theta[i])

            # k = 0
            # for j in range(lo, hi):
            #     d = abs(phi[i] - phi[j])
            #     if d < lcrphi:
            #         neighbors[k] = j
            #         k += 1

            neighbors_theta = np.arange(lo, hi)
            dists_phi = np.abs(phi[i] - phi[neighbors_theta])
            neighbors_phi = neighbors_theta[dists_phi < lcrphi]

            # n = 0
            # for j in range(k):
            #     neighbor = neighbors[j]
            #     d = np.sum(np.square(r[i] - r[neighbor]))
            #     if flux[neighbor] != 0 and d < self._critical:
            #         neighbors[n] = neighbor
            #         n += 1

            dists = np.sum(np.square(r[i] - r[neighbors_phi]), axis=1)
            ind_neighbors = (dists < self._critical) & (flux[neighbors_phi] != 0)
            neighbors = neighbors_phi[ind_neighbors]
            rng.shuffle(neighbors)



            if len(neighbors) > 1:
                flux_sum = np.sum(flux[neighbors])
                flux[neighbors[1:]] = 0.0
                flux[neighbors[0]] = flux_sum

        return self._collide_finish(phi, theta, flux, nflux)
