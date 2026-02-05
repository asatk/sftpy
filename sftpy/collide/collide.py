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
import numba as nb
import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..util import consolidate

dt = rc["general.dt"]
correction = rc["collide.correction"]
meanv = rc["collide.meanv"]
diffusion = rc["physics.diffusion"]
loglvl = rc["component.loglvl"]



@nb.jit(cache=True)
def calculate_pos(theta, phi, nflux):
    sintheta = np.sin(theta[:nflux])
    x = sintheta * np.cos(phi[:nflux])
    y = sintheta * np.sin(phi[:nflux])
    z = np.cos(theta[:nflux])
    l = (x, y, z)
    r = np.stack(l, axis=1)
    return r



@nb.jit(cache=True)
def collide2(phi, theta, flux, nflux, skips, crphi, order, seeds):
    sort_idx = np.argsort(theta[:nflux])
    phi[:nflux] = phi[sort_idx]
    theta[:nflux] = theta[sort_idx]
    flux[:nflux] = flux[sort_idx]

    neighbors_nz = flux[:nflux] != 0

    # r = calculate_pos(phi, theta, nflux)
    sintheta = np.sin(theta[:nflux])
    x = sintheta * np.cos(phi[:nflux])
    y = sintheta * np.sin(phi[:nflux])
    z = np.cos(theta[:nflux])
    l = (x, y, z)
    r = np.stack(l, axis=1)

    los = np.arange(0 - skips, nflux - skips, dtype=np.int64)
    los = np.mod(los, nflux)
    his = np.arange(skips, nflux + skips, dtype=np.int64)
    his = np.mod(his, nflux)

    for i in order:

        if not neighbors_nz[i]:
            continue

        lo = los[i]
        thetalo = theta[i] - crphi
        while theta[lo] > thetalo:
            lo = lo - skips
            if lo < 0:
                thetalo += 2 * np.pi
                lo += nflux

        his[lo] = i

        hi = his[i]
        thetahi = theta[i] + crphi
        while theta[hi] < thetahi:
            hi = hi + skips
            if hi >= nflux:
                thetahi -= 2 * np.pi
                hi -= nflux

        los[hi] = i

        lcrphi = crphi / sintheta[i]
        # this won't work if lo > hi
        if lo > hi:
            neighbors_theta = np.arange(lo, hi + nflux + 1, dtype=np.int64)
        else:
            neighbors_theta = np.arange(lo, hi + 1, dtype=np.int64)

        phi_diff = phi[i] - phi[neighbors_theta]
        phi_dist = np.abs(phi_diff)
        is_near_phi = phi_dist < lcrphi
        neighbors_phi = neighbors_theta[is_near_phi]

        r_diff = r[i] - r[neighbors_phi]
        r_dist = np.sum(np.square(r_diff), axis=1)
        is_near = r_dist < crphi ** 2
        neighbors = neighbors_phi[is_near]

        n_neighbors = len(neighbors)
        if n_neighbors > 1:
            np.random.seed(seeds[i])

            ind_coalesce = np.random.randint(len(neighbors))
            nbr_coalesce = neighbors[ind_coalesce]

            flux_sum = np.sum(flux[neighbors])

            flux[neighbors] = 0.0
            flux[nbr_coalesce] = flux_sum

            neighbors_nz[neighbors] = False
            neighbors_nz[nbr_coalesce] = True

    index = np.nonzero(flux[:nflux])[0]
    nnew = len(index)
    if nflux == nnew:
        return nflux

    phi[:nnew] = phi[index]
    theta[:nnew] = theta[index]
    flux[:nnew] = flux[index]

    return nnew



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
                 loglvl: int=loglvl):

        super().__init__(loglvl)
        self._dt = dt
        self._corr = correction
        # TODO same constant as in charges/charges
        self._difu = diffusion

        # collision param from schrijver+ 1997 l=1400km^2/s
        self._radius = 1400 / meanv * correction

        self._crphi = self._radius / 7e5

    @abc.abstractmethod
    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):
        ...



class COLNone(Collide):
    """
    No collisions between any flux spots.
    """

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux < 2:
            return nflux

        nnew = consolidate(phi, theta, nflux)
        self.log(2, f"spots remaining: {nnew}/{nflux}")
        return nnew



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
    Collide opposite-flux concentrations only.
    """

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux < 2:
            return nflux

        r = calculate_pos(theta, phi, nflux)

        # determine pos/neg concentrations
        indp = np.nonzero(np.sign(flux[:nflux]) == +1)[0]
        indn = np.nonzero(np.sign(flux[:nflux]) == -1)[0]

        self.log(1, f"npos {len(indp)} / {nflux}   nneg {len(indn)} / {nflux}")

        rn = r[indn]
        fluxn = flux[indn]

        # iterate through pos concentrations
        for i in indp:

            # determine neg neighbors to pos spot i
            nbrs = np.sum(np.square(r[i] - rn), axis=1) < self._crphi ** 2

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

        nnew = consolidate(phi, theta, flux, nflux)

        self.log(2, f"spots remaining: {nnew}/{nflux}")

        return nnew






class COL2(Collide):

    def __init__(self,
                 range: int=100,
                 dt: float = dt,
                 correction: float = correction,
                 meanv: float = meanv,
                 diffusion: float = diffusion,  # TODO check if diff is in IDL code
                 loglvl: int = loglvl):
        super().__init__(dt, correction, meanv, diffusion, loglvl)
        self._range = range

    @property
    def range(self):
        return self._range

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux < 2:
            return nflux

        skips = self._range
        # skips = nflux // 1000 + 1
        crphi = self._crphi
        order = rng.permutation(np.arange(nflux, dtype=np.int64))
        seeds = rng.integers(low=2 ** 64 - 1, size=nflux, dtype=np.uint64)

        nnew = collide2(phi, theta, flux, nflux, skips, crphi, order, seeds)

        self.log(2, f"spots remaining: {nnew}/{nflux}")

        return nnew
