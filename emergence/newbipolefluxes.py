import numpy as np
from ..constants import rng

def newbipolefluxes(newfluxflag,
                    newflux,
                    ntotal,
                    p,
                    binflux,
                    minflux,
                    maxflux,
                    iseed):
    newfluxflag = ntotal >= 0.5
    if newfluxflag == 0:
        return

    newflux = np.zeros(ntotal, dtype=np.int64)
    inew = 0

    ep = 1 / (1 - p)
    bf2 = binflux * 2
    mbinflux = maxflux / binflux

    new = np.asarray((p * rng.uniform(ntotal)**ep) + 0.5 / bf2, np.int64)
    retry = (new < minflux) | (new >= mbinflux)
    nretry = np.sum(retry)

    while nretry:
        new_retry = (p * rng.uniform(nretry)**ep) + 0.5) / bf2
        new[retry] = new_retry
        retry = (new < minflux) | (new >= mbinflux)
        nretry = np.sum(retry)
