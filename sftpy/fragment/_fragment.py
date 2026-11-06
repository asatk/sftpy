import numpy as np

from ..diffusion import ConstantRW
from ..sim import Actor, Component, Data, rng, Timeline
from ..sim.constants import BINFLUX

from .constants import MULTIPLIER, FRAG_DIST



class Fragment(Actor, Component):
    """
    `Fragment` donates flux from identified parent spots to new child spots.

    The child spot receives a random fraction less than half of the parent
    spot's original flux. The child undergoes a random walk over a small
    fragmentation distance that must be specified in the `RandomWalk` Component
    itself.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 frag_dist: float = FRAG_DIST,
                 multiplier: float = MULTIPLIER):

        super().__init__(time, data)

        diffusion_schedule = frag_dist ** 2 / 4 / time.dt_schedule()

        data_child = Data()
        self._data_child = data_child
        # create random walk component
        self._rwalk = ConstantRW(time, data_child, diffusion=diffusion_schedule)

        # k0 = 0.4e-24 Mx/s from Schrijver+ 97
        def _schedule_frag_prob():
            return 0.4e-6 * BINFLUX * multiplier * time.dt_schedule()

        self._get_frag_prob = time.create_schedule(_schedule_frag_prob)

    def simulate(self):
        """
        Randomly split spots into two spots of an equal combined absolute flux
        with probability dependent on the absolute flux of the original spot.
        """

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        # magnitude of each spot's flux
        aflux = np.abs(flux[:nflux])

        # old prob/lifetime had + 0.008 not 0.12
        # see ref Schrijver+ 1997c (SPh)
        # flux-dependent break-up probability of spots
        kdt = self._get_frag_prob()
        prob = aflux * kdt * (np.exp(-aflux * BINFLUX / 120.) + 0.12)

        # bernoulli trial determining which spots will fragment into 2 children
        parents = np.nonzero(rng.uniform(size=nflux) < prob)[0]
        nparents = len(parents)

        # no spots selected for fragmentation
        if nparents == 0:
            return

        # fraction of parent's flux to become a new spot
        aflux_child = rng.uniform(size=nparents) * aflux[parents]
        aflux_child = np.astype(aflux_child, np.int64)
        # aflux_child = rng.integers(low=0, high=aflux[parents], endpoint=False)
        aflux_child = np.min([aflux_child, aflux[parents] - aflux_child], axis=0)
        flux_child = np.sign(flux[parents]) * aflux_child

        # child spots must have non-zero flux
        has_flux = flux_child != 0
        if not np.any(has_flux):
            return

        # fragment only parents with non-zero child fluxes
        parents = parents[has_flux]
        flux_child = flux_child[has_flux]
        nchild = len(parents)

        # subtract child's flux from parent's flux
        flux[parents] -= flux_child

        # create new arrays -- specific locs in mem for diffusion to write into
        phi_child = phi[parents].copy()
        theta_child = theta[parents].copy()

        # modify data object for fragmented spots that take a random walk
        data_child = self._data_child
        data_child.insert_sources(phi_child, theta_child, flux_child, nchild)

        # move child fragments over some fragmentation distance
        self._rwalk.simulate()

        # add child fragments to list of all sources
        self._data.append_data(data_child)
