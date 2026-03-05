import numpy as np

from sftpy import simrc as rc
from sftpy import rng

from ..component import Component
from ..util import consolidate

dt = rc["general.dt"]
t_decay = rc["decay.t_decay"]
loglvl = rc["component.loglvl"]

class Decay(Component):
    """
    `Decay` gradually reduces the overall stellar magnetic flux using an
    exponential decay with a characteristic timescale.
    """

    prefix = "[decay]"

    def __init__(self,
                 dt: float=dt,
                 t_decay: float=t_decay,
                 loglvl: int=loglvl):
        """
        Create a `Decay` component with a timestep and characteristic decay
        time.

        The timescales `dt` and `t_decay` determine the factor by which total
        absolute magnetic flux reduces. A decay constant `t_decay` greater than
        999 yrs yields no decay.

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
        if t_decay > 999.0:
            self.log(0, "long decay half-life (> 999 yr): no decay")
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
        Remove equal parts positive and negative flux from the surface field.

        One unit of flux is removed from spots chosen at random. The total
        absolute flux is decreased by a factor of $1 - 2^{dt/t_{decay}}$.

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
        if rng.uniform() < (remove - ndecay):
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

        # number of spots limited to at most those available in each polarity
        ndecay_pos = min(ndecay, len(pos))
        ndecay_neg = min(ndecay, len(neg))

        # select `ndecay` spots to lose 1 flux unit (random w/o replacement)
        ind_pos_decay = rng.choice(pos, size=ndecay_pos, replace=False)
        ind_neg_decay = rng.choice(neg, size=ndecay_neg, replace=False)

        # indices of all spots that will decay flux
        ind_decay = np.r_[ind_pos_decay, ind_neg_decay]

        # decay the spots chosen to lose flux
        flux[ind_decay] -= np.sign(flux[ind_decay])

        # remove empty spots
        nremain = consolidate(phi, theta, flux, nflux)
        return nremain
