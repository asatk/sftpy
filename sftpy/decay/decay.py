import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component

dt = rc["general.dt"]
t_decay = rc["decay.t_decay"]
loglvl = rc["component.loglvl"]

class Decay(Component):
    """
    `Decay` gradually reduces the overall stellar magnetic flux according to
    patterns observed on the Sun.

    REF

    """

    prefix = "[decay]"

    def __init__(self,
                 dt: float=dt,
                 t_decay: float=t_decay,
                 loglvl: int=loglvl):
        """
        Create a Decay component, optionally specifying the timestep and decay
        time.

        The timescales `dt` and `t_decay` determine the factor by which total
        absolute magnetic flux reduces. A decay constant `t_decay` of 1000 yrs
        or greater yields no decay.

        Parameters
        ----------
        dt : float
            Timestep (seconds).
        t_decay : float
            Decay constant (years).
        loglvl : int
            Logging level for this component.

        Notes
        -----
        The factor by which the total absolute flux is reduced is $1 - 2^{
        -dt/t_decay}$. From each polarity, half of this amount of flux is
        removed from spots randomly chosen with replacement.

        REFS
        """
        super().__init__(loglvl)

        # no decay if decay timescale is too large
        if t_decay >= 1000:
            self.log(0, "long decay half-life (> 1000 yr): no decay")
            self._factor = 0
        else:
            self._factor = 1 - np.exp(
                -np.log(2) * dt / 365.25 / 86400 / t_decay)


    def decay(self, 
              phi: np.ndarray,
              theta: np.ndarray,
              flux: np.ndarray,
              nflux: int):
        """
        Remove flux from the total field.

        Flux is removed equally from both positive and negative spots. Spots
        are chosen at random. The decrement is determined by the factor
        $1 - 2^{dt/t_{decay}}$.

        Parameters
        ----------

        phi : np.ndarray
            Array of longitudes of spots (radians)
        theta : np.ndarray
            Array of co-latitudes of spots (radians)
        flux : np.ndarray
            Array of fluxes of spots (int)
        nflux : int
            Number of spots

        Returns
        -------
        nflux : int
            Number of spots remaining after decay

        """

        # too small of a timescale to decay any magnetic flux
        if self._factor == 0:
            return nflux

        # total absolute flux of all spots
        aflux = np.sum(np.abs(flux))

        # flux to be removed per polarity (hence div by 2)
        remove = aflux / 2 * self._factor

        # integer amount of flux to decay
        ndecay = int(remove)

        # random chance to round up fractional flux amount to integer
        if rng.uniform() < remove - ndecay:
            ndecay += 1

        self.log(1, f"remove flux per polarity: {ndecay}/{aflux}")

        # no flux will be removed
        if ndecay == 0:
            return nflux

        # no non-zero fluxes exist (only empty spots)
        if np.count_nonzero(flux[:nflux]) == 0:
            self.log(0, f"no spots remaining: {nflux} -> 0")
            return 0

        # identify all positive and negative spots
        pos = np.nonzero(flux[:nflux] > 0)[0]
        neg = np.nonzero(flux[:nflux] < 0)[0]

        # select `ndecay` spots from which to decay flux (random w/ replacement)
        ind_pos_decay = rng.choice(pos, size=ndecay, replace=True)
        ind_neg_decay = rng.choice(neg, size=ndecay, replace=True)

        # indices of all spots that will decay flux
        ind_decay = np.r_[ind_pos_decay, ind_neg_decay]

        # determine the unique indices of spots to be decayed and their counts
        ind_unique, flux_decay = np.unique(ind_decay, return_counts=True)

        # determine spots that will have no flux after decaying
        will_empty = flux[ind_unique] <= flux_decay

        # decay only the flux they have available -- don't want to flip signs
        flux_decay[will_empty] = flux[ind_unique][will_empty]

        # decay the spots chosen to lose flux
        flux[ind_unique] -= np.sign(flux[ind_unique]) * flux_decay

        # no empty spots -- move on to next step
        if not np.any(will_empty):
            return nflux

        # remove empty spots
        ind = np.nonzero(flux[:nflux])[0]
        nremain = len(ind)

        # only maintain list spots with non-zero flux
        phi[:nremain] = phi[ind]
        theta[:nremain] = theta[ind]
        flux[:nremain] = flux[ind]

        return nremain