"""
Decay field
"""

import numpy as np

def field_decay(theta,
                phi,
                flux: np.ndarray,
                rng,
                nflux: int,
                dt: float,
                decay_t: float):

    # no decay if decay timescale is too large
    if decay_t > 999:
        return

    # number of flux concentrations to remove for this step
    remove = 1 - np.exp(-np.log(2) * dt / 365.25 / 86400 / decay_t)
    remove *= np.sum(np.abs(flux) / 2)

    # fractional amt rounded up w/ random prob equal to fraction
    remainder = remove - np.int64(remove)
    if rng.uniform() < rest:
        remove += 1

    # integer number of flux concentrations to remove
    remove = np.int64(remove)

    # no flux concentrations will be removed
    if remove == 0:
        return

    # identify all positive and negative flux concentrations
    pos = np.nonzero(flux[:nflux] > 0)[0]
    neg = np.nonzero(flux[:nflux] < 0)[0]

    # TODO should this be pos < 0 and neg > 0 -- i don't understand this stmt
    if pos[0] < 0 or neg[0] < 0:
        return

    # select `remove` number of positive and negative concentrations to decay
    pos_decay = np.choice(pos, size=remove, replace=False)
    neg_decay = np.choice(neg, size=remove, replace=False)

    flux[pos_decay] -= 1
    flux[neg_decay] += 1

    # "remove 'empty' concentrations if necessary"
    # TODO how can we avoid this -- array list?
    ind = flux[:nflux] != 0
    ndecay = np.sum(ind)
    if ndecay == nflux:
        return

    nflux = ndecay

    phi[:nflux] = phi[:nflux][ind]
    theta[:nflux] = theta[:nflux][ind]
    flux[:nflux] = flux[:nflux][ind]

    return nflux

