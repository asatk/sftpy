"""
Decay field
"""

import numpy as np

class Decay():
    """
    Component class for random decay of flux conecentrations.
    """

    def __init__(self,
                 dt: float,
                 rng: np.random.Generator,
                 t_decay: float):
        self._rng = rng
        self._factor = 1 - np.exp(-np.log(2) * dt / 365.25 / 86400 / t_decay)

        # no decay if decay timescale is too large
        if t_decay > 999:
            print("bruh that's too long")
            return None


    def decay(self, 
              phi: np.ndarray,
              theta: np.ndarray,
              flux: np.ndarray,
              nflux: int):

        # number of flux concentrations to remove for this step
        remove = np.sum(np.abs(flux) / 2) * self._factor

        # fractional amt rounded up w/ random prob equal to fraction
        remainder = remove - np.int64(remove)
        if self._rng.uniform() < remainder:
            remove += 1

        # integer number of flux concentrations to remove
        remove = np.int64(remove)

        print(f"[decay] ---- remove: {remove}/{nflux}")

        # no flux concentrations will be removed
        if remove == 0:
            return nflux

        # identify all positive and negative flux concentrations
        pos = np.nonzero(flux[:nflux] > 0)[0]
        neg = np.nonzero(flux[:nflux] < 0)[0]

        # TODO should this be pos < 0 and neg > 0 -- i don't understand this stmt
        if not np.any(pos) and not np.any(neg):
            return nflux

        # select `remove` number of positive and negative concentrations to decay
        pos_decay = self._rng.choice(pos, size=remove, replace=False)
        neg_decay = self._rng.choice(neg, size=remove, replace=False)

        flux[pos_decay] -= 1
        flux[neg_decay] += 1

        # "remove 'empty' concentrations if necessary"
        # so this takes care of concentrations that have been fully decayed
        # TODO how can we avoid this -- array list?
        ind = flux[:nflux] != 0
        ndecay = np.sum(ind)

        if ndecay == nflux:
            return nflux


        phi[:ndecay] = phi[:nflux][ind]
        theta[:ndecay] = theta[:nflux][ind]
        flux[:ndecay] = flux[:nflux][ind]

        nflux = ndecay

        return nflux

