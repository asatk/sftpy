"""
Collision schemes from the original Schrijver+ model:
 - COL1: sequentially coalesce nearby spots into randomly-selected final spot.
 - COL2: ?
 - COL3: like COL2 but in C

"""


# parameter in kit_iocontrol.pro `collide`
# Collisions 0-none 1-opposite polarity, 2-both polarities

import abc
import numba as nb
import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..util import consolidate
from ..util.other import phithetaxyz

dt = rc["general.dt"]
correction = rc["collide.correction"]
meanv = rc["collide.meanv"]
diffusion = rc["physics.diffusion"]
loglvl = rc["component.loglvl"]
rad = rc["physics.rad"]





@nb.jit(cache=True)
def collide2(phi, theta, flux, nflux, skips, crphi, order, rvs):
    sort_idx = np.argsort(theta)

    ### > COMPARE PERFORMANCE

    # phi[:] = phi[sort_idx]
    # theta[:] = theta[sort_idx]
    # flux[:] = flux[sort_idx]

    phi = phi[sort_idx]
    theta = theta[sort_idx]
    flux = flux[sort_idx]

    ### <

    neighbors_nz = flux != 0

    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    l = (x, y, z)
    r = np.stack(l, axis=1)

    los = np.arange(0 - skips, nflux - skips, dtype=np.int64)
    los = np.clip(los, a_min=0, a_max=nflux-1)
    # los = np.mod(los, nflux)

    his = np.arange(skips, nflux + skips, dtype=np.int64)
    his = np.clip(his, a_min=0, a_max=nflux-1)
    # his = np.mod(his, nflux)

    # calculate finding neighbors in parallel and coalesce sequentially?
    for i in order:

        # print(f"--- {i = }")

        # print(f"{neighbors_nz = })")

        # flux of concentration must be non-zero
        # if not flux[i]:
        #     continue

        if not neighbors_nz[i]:
            continue

        lo = los[i]
        thetalo = theta[i] - crphi
        # print(f"{i = } {thetalo = :.3f}")
        while theta[lo] > thetalo:
            # print(f"{lo = } {theta[lo] = :.3f}")
            lo = lo - skips
            if lo < 0:
                thetalo += 2 * np.pi
                lo += nflux

        if lo > i:
            lo -= nflux

        # his[lo] = i

        hi = his[i]
        thetahi = theta[i] + crphi
        # print(f"{i = } {thetahi = :.3f}")
        while theta[hi] < thetahi:
            # print(f"{hi = } {theta[hi] = :.3f}")
            hi = hi + skips
            if hi >= nflux:
                thetahi -= 2 * np.pi
                hi -= nflux

        if hi < i:
            hi += nflux

        # los[hi] = i

        lcrphi = crphi / sintheta[i]

        # if lo > hi:
        #     neighbors_theta = np.arange(lo, hi + nflux + 1, dtype=np.int64)
        #     neighbors_theta = np.mod(neighbors_theta, nflux)
        # else:

        neighbors_theta = np.arange(lo, hi + 1, dtype=np.int64) % nflux

        # make sure spots that are marked as empty never get looked at again
        # neighbors_theta_nz = neighbors_theta[neighbors_nz[neighbors_theta]]
        # hm accessing neighbors_nz in this way doubles comp time...

        # print(f"{lo = } {hi = }")
        # print(f"{neighbors_theta = }")


        phi_diff = phi[i] - phi[neighbors_theta]
        # phi_diff = phi[i] - phi[neighbors_theta_nz]
        phi_dist = np.abs(phi_diff)
        is_near_phi = phi_dist < lcrphi
        neighbors_phi = neighbors_theta[is_near_phi]
        # neighbors_phi = neighbors_theta_nz[is_near_phi]

        r_diff = r[i] - r[neighbors_phi]
        r_dist = np.sum(np.square(r_diff), axis=1)
        is_near = r_dist < (crphi ** 2)
        # neighbors = neighbors_phi[is_near]

        neighbors_temp = neighbors_phi[is_near]
        neighbors = np.unique(neighbors_temp[flux[neighbors_temp] != 0])

        n_neighbors = len(neighbors)
        if n_neighbors > 1:

            # np.random.seed(seeds[i])
            # ind_coalesce = np.random.randint(n_neighbors)
            ind_coalesce = int(rvs[i] * n_neighbors)
            nbr_coalesce = neighbors[ind_coalesce]

            flux_sum = np.sum(flux[neighbors])

            flux[neighbors] = 0
            neighbors_nz[neighbors] = False

            if flux_sum != 0:
                flux[nbr_coalesce] = flux_sum
                neighbors_nz[nbr_coalesce] = True

    index = np.nonzero(flux)[0]
    nnew = len(index)

    if nnew < nflux:
        phi[:nnew] = phi[index]
        theta[:nnew] = theta[index]
        flux[:nnew] = flux[index]

    phi_r = phi[:nnew]
    theta_r = theta[:nnew]
    flux_r = flux[:nnew]

    l_ret = (phi_r, theta_r, flux_r)

    return l_ret


# @nb.jit(cache=True)
def collide_idl(phi, theta, flux, nflux, skips, crphi, order, seeds):

    mlen = 10000
    index = np.zeros(mlen, dtype=np.int64)

    skips = 25

    nlen = phi.shape[0]
    x = np.zeros(nlen)
    y = np.zeros(nlen)
    z = np.zeros(nlen)

    nflux1 = nflux - 1

    index_sort = np.argsort(theta[:nflux])
    theta[:nflux] = theta[index_sort]
    phi[:nflux] = phi[index_sort]
    flux[:nflux] = flux[index_sort]

    for i in range(nflux):
        sintheta = np.sin(theta[i])
        x[i] = sintheta * np.cos(phi[i])
        y[i] = sintheta * np.sin(phi[i])
        z[i] = np.cos(theta[i])

    for i in range(nflux - 2):
        ipick = order[i]
        if flux[ipick] == 0:
            continue

        thetamin = theta[ipick] - crphi
        low = ipick - skips
        if low < 0:
            low = 0

        while (theta[low] > thetamin) and (low > 0):
            low = low - skips
            if low < 0:
                low = 0

        thetaplus = theta[ipick] + crphi
        hih = ipick + skips
        if hih > nflux1:
            hih = nflux1

        while (theta[hih] < thetaplus) and (hih < nflux1):
            hih = hih + skips
            if hih > nflux1:
                hih = nflux1

        lcrphi = crphi / np.sin(theta[ipick])

        k = 0
        for j in range(low, hih):
            d = phi[ipick] - phi[j]
            if d < 0:
                d = -d
            if d < lcrphi:
                # print(i, j, k, d, lcrphi)
                index[k] = j
                k += 1


        nii = 0
        for j in range(k):
            ij = index[j]
            d = (x[ipick] - x[ij])**2 + \
                (y[ipick] - y[ij])**2 + \
                (z[ipick] - z[ij])**2
            if (d < crphi**2) and (flux[ij] != 0):
                index[nii] = ij
                nii += 1

        # no neighbors
        if nii == 0:
            continue

        np.random.seed(seeds[i])
        ic = np.random.randint(nii)
        ic = index[ic]

        for j in range(nii):
            if ic != index[j]:
                ij = index[j]
                flux[ic] = flux[ic] + flux[ij]
                flux[ij] = 0

    nfluxold = nflux
    index = np.nonzero(flux[:nflux])[0]
    # print(index)
    nflux = len(index)
    if nfluxold == nflux:
        return nflux

    phi[:nflux] = phi[index]
    theta[:nflux] = theta[index]
    flux[:nflux] = flux[index]

    return nflux



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
        self._radius = 1400. / meanv * correction
        # npix = 2
        # self._radius = npix * rad * np.pi / 180.0

        self._crphi = self._radius / rad

    @abc.abstractmethod
    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):
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

        r = phithetaxyz(phi, theta, nflux)

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

    @range.setter
    def range(self, value):
        self._range = value

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if self._loglvl > 1:
            fluxtot_pre = np.sum(np.abs(flux[:nflux]))

        # number of indices that can be skipped must be no more than the total number of spots
        skips = 1 if nflux <= self._range else self._range
        crphi = self._crphi
        order = rng.permutation(np.arange(nflux, dtype=np.int64))
        # seeds = rng.integers(low=2 ** 32 - 1, size=nflux, dtype=np.uint32)
        rvs = rng.uniform(size=nflux)

        phi_in = phi[:nflux]
        theta_in = theta[:nflux]
        flux_in = flux[:nflux]

        phi_out, theta_out, flux_out = collide2(phi_in, theta_in, flux_in, nflux, skips, crphi, order, rvs)
        nnew = phi_out.shape[0]

        phi[:nnew] = phi_out
        theta[:nnew] = theta_out
        flux[:nnew] = flux_out

        # unnecessary but keeps the data clean
        phi[nnew:] = 0.0
        theta[nnew:] = 0.0
        flux[nnew:] = 0

        if self._loglvl > 1:
            fluxtot_post = np.sum(np.abs(flux[:nnew]))
            self.log(1, f"\tdelta nflux: {nnew-nflux:+6d} / {nflux:6d}\t" + \
                 f"delta flux: {fluxtot_post-fluxtot_pre:7d} / {fluxtot_pre:7d}")

        return nnew
