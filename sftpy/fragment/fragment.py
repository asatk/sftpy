import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..rwalk import RandomWalk

dt = rc["general.dt"]
mult = rc["fragment.mult"]
correction = rc["fragment.correction"]
loglvl = rc["component.loglvl"]
binflux = rc["physics.binflux"]


class Fragment(Component):
    """
    `Fragment` reduces the flux in a parent spot by giving a portion (less
    than half) of its flux to a child spot.

    The child spot will always undergo a random walk over a small
    fragmentation distance that is specified in the `RandomWalk` component
    itself.
    """

    def __init__(self,
                 rwalk: RandomWalk,
                 dt: float=dt,
                 mult: float=mult,
                 correction: float=correction,
                 loglvl: int=loglvl):
        """
        Create a `Fragment` component, optionally specifying the timestep,
        relative fragmentation strength, and correction to collision frequency.

        Parameters
        ----------
        rwalk : RandomWalk
            `RandomWalk` component that handles the random motion of fragments
            over a distance after fragmenting.
        dt : float
            Timestep (s).
        mult : float
            Strength of fragmentation (rel.).
        correction : float
            Correction factor for collision frequency (rel.).
        loglvl : int
            Maximum level for logging
        """
        super().__init__(loglvl=loglvl)
        self._rwalk = rwalk
        self._dt = dt
        # k0 = 0.4e-24 Mx/s from Schrijver+ 97
        self._k0 = 0.4e-6 * binflux * mult * correction

    def fragment(self,
                 phi: np.ndarray,
                 theta: np.ndarray,
                 flux: np.ndarray,
                 nflux: int):
        """
        Randomly split spots into two spots of an equal combined absolute flux
        with probability dependent on the absolute flux of the original spot.

        Parameters
        ----------
        phi : np.ndarray
            Longitudes of spots
        theta : np.ndarray
            Co-latitudes of spots
        flux : np.ndarray
            Fluxes of spots
        nflux : int
            Number of spots

        Returns
        -------
        nflux : int
            Number of spots after fragmenting
        """

        # magnitude of each spot's flux
        aflux = np.abs(flux[:nflux])

        # old prob/lifetime had + 0.008 not 0.12
        # see ref Schrijver+ 1997c (SPh)
        # flux-dependent break-up probability of spots
        prob = ((aflux * np.exp(-aflux * binflux / 120.) + 0.12)
                * self._k0
                * self._dt)

        # bernoulli trial determining which spots will fragment into 2 children
        parents = np.nonzero(rng.uniform(size=nflux) < prob)[0]
        nparents = len(parents)

        # no spots selected for fragmentation
        if nparents == 0:
            return nflux

        # fraction of parent's flux to become a new spot
        flux_child = rng.uniform(high=0.5, size=nparents) * flux[parents]
        flux_child = np.astype(flux_child, np.int64)

        # child spots must have non-zero flux
        has_flux = flux_child > 0
        if not np.any(has_flux):
            return nflux

        # fragment only parents with non-zero child fluxes
        parents = parents[has_flux]
        flux_child = flux_child[has_flux]
        nchild = len(parents)

        # subtract child's flux from parent's flux
        flux[parents] -= flux_child

        # create new arrays -- specific locs in mem for rwalk to write into
        phi_child = phi[parents].copy()
        theta_child = theta[parents].copy()
        
        # move child fragments over some fragmentation distance
        self._rwalk.move(phi_child, theta_child, flux_child, nchild)

        # add child fragments to list of spots
        phi[nflux:nflux+nchild] = phi_child
        theta[nflux:nflux+nchild] = theta_child
        flux[nflux:nflux+nchild] = flux_child
        nflux += nchild

        return nflux
