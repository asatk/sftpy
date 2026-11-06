import numpy as np

from ..sim import Actor, Component, Data, rng, Timeline
from ..util import consolidate

from .constants import T_DECAY



class SchrijverDecay(Actor, Component):
    """
    `Decay` gradually reduces the overall stellar magnetic flux using an
    exponential decay with a characteristic timescale.
    """

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 t_decay: float = T_DECAY):
        """
        Create a `Decay` _abcs with a timestep and characteristic decay
        time.

        The timescales `dt` and `t_decay` determine the factor by which total
        absolute magnetic flux reduces. A decay constant `t_decay` greater than
        999 yrs yields no decay.

        Parameters
        ----------
        t_decay : float
            Decay constant (years).
        logging : int
            Logging level for this Component.

        Notes
        -----
        The factor by which the total absolute flux is reduced is $1 - 2^{
        -dt/t_decay}$. From each polarity, half of this amount of flux is
        removed from spots randomly chosen with replacement.

        REFS
        """
        super().__init__(time, data)

        def _schedule_decay_factor():
            return 1 - np.exp(-np.log(2) * time.time_schedule("y") / t_decay)

        self._get_decay_factor = time.create_schedule(_schedule_decay_factor())

    def simulate(self):
        """
        Remove equal parts positive and negative flux from the surface field.

        One unit of flux is removed from spots chosen at random. The total
        absolute flux is decreased by a factor of $1 - 2^{dt/t_{decay}}$.
        """

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        factor = self._get_decay_factor()

        # total absolute flux of all spots
        aflux = np.sum(np.abs(flux))

        # flux to be removed per polarity (hence div by 2)
        remove = aflux / 2 * factor

        # integer amount of flux to decay
        ndecay = int(remove)

        # random chance to round up fractional flux amount to integer
        if rng.uniform() < (remove - ndecay):
            ndecay += 1

        self.log(1, f"remove flux per polarity: {ndecay}/{aflux}")

        # no flux will be removed
        if ndecay == 0:
            return

        # no non-zero fluxes exist (only empty spots)
        if np.count_nonzero(flux[:nflux]) == 0:
            self.log(1, f"no spots remaining: {nflux} -> 0")
            # remove empty spots
            nremain = consolidate(phi, theta, flux, nflux)
            data.nflux = nremain
            return

        # identify all positive and negative spots
        pos = np.asarray(flux[:nflux] > 0).nonzero()[0]
        neg = np.asarray(flux[:nflux] < 0).nonzero()[0]

        # number of spots limited to at most those available in each polarity
        ndecay_pos = min(ndecay, len(pos))
        ndecay_neg = min(ndecay, len(neg))

        # select `ndecay` spots to lose 1 flux unit (random w/o replacement)
        ind_pos_decay = pos[np.argsort(rng.uniform(size=ndecay_pos))]
        ind_neg_decay = neg[np.argsort(rng.uniform(size=ndecay_neg))]

        # indices of all spots that will decay flux
        ind_decay = np.r_[ind_pos_decay, ind_neg_decay]

        # decay the spots chosen to lose flux
        flux[ind_decay] -= np.sign(flux[ind_decay])

        # remove empty spots
        nremain = consolidate(phi, theta, flux, nflux)
        data.nflux = nremain
