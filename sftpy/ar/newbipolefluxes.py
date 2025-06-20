import numpy as np

def new_bipole_fluxes(ntotal: int, p: float, binflux: float, minflux: float,
                      maxflux: float, rng):

    newflux = np.zeros(ntotal, dtype=np.int64)

    ep = 1 / (1.0 - p)
    bf2 = binflux * 2
    mbinflux = maxflux / binflux

    newvals = np.asarray((p * rng.uniform(size=ntotal) ** ep + 0.5) / bf2,
                         np.int64)

    notvalid = (newvals < minflux) & (newvals >= mbinflux)
    while np.any(notvalid):
        nreplace = len(notvalid)
        replacevals = np.asarray((p * rng.uniform(size=nreplace) ** ep + 0.5) / bf2,
                         np.int64)
        newvals[notvalid] = replacevals
        notvalid = (newvals < minflux) & (newvals >= mbinflux)

    return newvals
