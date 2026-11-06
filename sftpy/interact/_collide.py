"""
Collision schemes from the original Schrijver+ model:
 - COL1: sequentially coalesce nearby spots into randomly-selected final spot.
 - COL2: ?
 - COL3: like COL2 but in C

"""

import numba as nb
import numpy as np
import numpy.typing as npt

from ..sim import Actor, Component, Data, rng, Timeline
from ..sim.constants import STAR_RADIUS

from .constants import NSKIP, COL_RADIUS



@nb.jit(cache=True)
def collide2(phi: npt.NDArray[np.float64],
             theta: npt.NDArray[np.float64],
             flux: npt.NDArray[np.int64],
             nflux: int,
             skips: int,
             crphi: float,
             order: npt.NDArray[np.int64],
             rvs: npt.NDArray[np.float64]):

    sort_idx = np.argsort(theta)

    phi[:] = phi[sort_idx]
    theta[:] = theta[sort_idx]
    flux[:] = flux[sort_idx]

    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    l = (x, y, z)
    r = np.stack(l, axis=1)

    los = np.arange(0 - skips, nflux - skips, dtype=np.int64)
    # los = np.clip(los, a_min=0, a_max=nflux-1)
    # los = np.mod(los, nflux)

    his = np.arange(skips, nflux + skips, dtype=np.int64)
    # his = np.clip(his, a_min=0, a_max=nflux-1)
    # his = np.mod(his, nflux)

    # calculate finding neighbors in parallel and coalesce sequentially?
    for i in order:

        # flux of concentration must be non-zero
        if not flux[i]:
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

        # latitude-corrected critical angular distance
        lcrphi = crphi / sintheta[i]

        # identify indices of spots within critical angular distance in co-lat
        neighbors_theta = np.arange(lo, hi + 1, dtype=np.int64) % nflux

        phi_diff = phi[i] - phi[neighbors_theta]
        phi_dist = np.abs(phi_diff)
        is_near_phi = phi_dist < lcrphi
        neighbors_phi = neighbors_theta[is_near_phi]

        r_diff = r[i] - r[neighbors_phi]
        r_dist = np.sum(np.square(r_diff), axis=1)
        is_near = r_dist < (crphi ** 2)

        # make sure spots that are marked as empty never get looked at again
        neighbors_temp = neighbors_phi[is_near]

        # only consider neighbors with non-zero flux
        neighbors = neighbors_temp[flux[neighbors_temp] != 0]

        n_neighbors = len(neighbors)
        if n_neighbors > 1:

            ind_coalesce = int(rvs[i] * n_neighbors)
            nbr_coalesce = neighbors[ind_coalesce]

            flux_sum = np.sum(flux[neighbors])

            flux[neighbors] = 0

            if flux_sum != 0:
                flux[nbr_coalesce] = flux_sum

    index = np.nonzero(flux)[0]
    nnew = len(index)

    phi[:nnew] = phi[index]
    theta[:nnew] = theta[index]
    flux[:nnew] = flux[index]

    return nnew



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



class SchrijverCollide(Actor, Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 collide_radius: float= COL_RADIUS,
                 nskip: int = NSKIP):
        super().__init__(time, data)
        self._col_radius = collide_radius
        self._nskip = nskip
        self._crphi = collide_radius / STAR_RADIUS

    def simulate(self):

        data = self._data
        nflux = data.nflux
        phi_in = data.phi[:nflux]
        theta_in = data.theta[:nflux]
        flux_in = data.flux[:nflux]

        # number of indices that can be skipped must be no more than the total number of spots
        nskip = 1 if nflux <= self._nskip else self._nskip
        crphi = self._crphi
        order = rng.permutation(np.arange(nflux, dtype=np.int64))
        rvs = rng.uniform(size=nflux)

        nnew = collide2(phi_in, theta_in, flux_in, nflux, nskip, crphi, order, rvs)

        data.clear(n=nnew)
        # data.append_sources(phi_out, theta_out, flux_out, nnew)
