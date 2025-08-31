import abc
import numpy as np

from ..component import Component

class Collide(Component, metaclass=abc.ABCMeta):
    """
    Base class for flux concentration collision component of computation sequence.
    """

    meanv: float = 1 / 3    # estimate of mean velocity
    prefix = "[collide]"

    def __init__(self,
                 dt: float,
                 rng: np.random.Generator,
                 correction: float=1.0,
                 diffusion: float=300.0,    # TODO check if diff is in IDL code
                 trackcancel: bool=False,
                 loglvl: int=0):
        super().__init__(loglvl)
        self._dt = dt
        self._rng = rng
        self._corr = correction
        # TODO same constant as in charges/charges
        self._difu = diffusion
        self._track = trackcancel

        # collision param from schrijver+ 1997 l=1400km^2/s
        self._radius = 1400 / self.meanv * correction

        self._crphi = self._radius / 7e5
        self._critical = self._crphi ** 2 # collision distance in units of stellar radius
        self._cr = int(3 * self._radius * 128 / 700000 + 0.5)

    def _collide_start(self,
                       phi: np.ndarray,
                       theta: np.ndarray,
                       flux: np.ndarray,
                       nflux: int):

        # TODO shouldnt theta already be in this range?
        # TODO this will be implemented via normalization components
        # TODO wait this is weird... periodic BCs for the poles? unphysical
        theta_mod = (theta[:nflux] + np.pi) % np.pi
        
        ind = np.argsort(theta_mod)
        theta[:nflux] = theta[ind]
        phi[:nflux] = phi[ind]
        flux[:nflux] = flux[ind]


        # coordinates
        sintheta = np.sin(theta[:nflux])
        x = sintheta * np.cos(phi[:nflux])
        y = sintheta * np.sin(phi[:nflux])
        z = np.cos(theta[:nflux])
        r = np.stack([x, y, z])
        rb = np.trunc((r + 1) * 128).astype(np.byte)

        return r, rb

    def _collide_finish(self,
                        phi: np.ndarray,
                        theta: np.ndarray,
                        flux: np.ndarray,
                        nflux: int):

        nfluxold = nflux
        index = np.nonzero(flux[:nflux] != 0)[0]
        nnew = len(index)
        if nflux == nnew:
            return nflux

        self.log(2, f"[collide finish] ---- nnew {nnew}/{nflux}")
        self.log(2, f"[collide finish] ---- index {index}")
        nflux = nnew
        phi[:nnew] = phi[index]
        phi[:nnew] = theta[index]
        flux[:nnew] = flux[index]

        return nflux

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

        r, rb = self._collide_start(phi, theta, flux, nflux)
        return self._collide_finish(phi, theta, flux, nflux)


class COL1(Collide):

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        r, rb = self._collide_start(phi, theta, flux, nflux)

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

            """
            ind1 = np.nonzero(
                    (np.sign(flux[i]) != np.sign(flux[i+1:nflux])) & \
                    (flux[i+1:nflux] != 0))[0] + i+1

            # TODO switch order of dims for r/rb
            ind2 = np.nonzero(
                    (np.sum(np.abs(r[:,i,None] - r[:,ind1]), axis=0) < self._cr) & \
                    (flux[i+1:nflux] != 0))[0] + i+1
            """

            if np.any(ind2):
                ind3 = np.nonzero(
                        np.sum(np.square(r[:,i,None] - r[:,ind2]), axis=0) < self._critical)[0]

                if np.any(ind3):
                    flux[i] += np.sum(flux[ind2[ind3]])
                    flux[ind2[ind3]] = 0    # eliminate collided particle(s)

                    # TODO there are some -1s in the IDL... compare code snippets
                    ic = self._rng.choice(ind3)
                    # TODO check this is just a not empty condition
                    if ic > 0:
                        phi[i] = phi[ind2[ic]]
                        theta[i] = theta[ind2[ic]]

        return self._collide_finish(phi, theta, flux, nflux)


class COL2(Collide):

    def collide(self,
                phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

        if nflux <= 2:
            return nflux

        r, rb = self._collide_start(phi, theta, flux, nflux)

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

                ic = self._rng.choice(ind2)
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

        r, rb = self._collide_start(phi, theta, flux, nflux)

        # TODO compiled C code -- just write in py for now
        # TODO um
        nflux = long(nflux)
        # TODO but this isnt nflux?
        #order = np.zeros(len(flux))

        order = self._rng.permutation(np.arange(nflux, dtype=np.int64))

        for i in range(nflux-1):
            
            ind = order[i]
            
            if flux[ind] == 0:
                continue

            thetalo = theta[ind] - self._crphi
            lo = ind-25
            if lo < 0:
                lo = 0

            while (theta[lo] > thetalo and low > 0):
                lo = lo - range
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
            ic = self._rng.integers(n)
            ic = index[ic]

            for j in range(n):
                if ic != index[j]:
                    ind2 = index[j]
                    flux[ic] += flux[ind2]
                    flux[ind2] = 0

        return self._collide_finish(phi, theta, flux, nflux)
