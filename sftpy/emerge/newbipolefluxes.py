import numpy as np

def new_bipole_fluxes(ntotal: int, p: float, binflux: float, minflux: float,
                      maxflux: float, rng):

    newflux = np.zeros(ntotal, dtype=np.int64)
    if ntotal == 0:
        return newflux

    print(f"[nbf] ---- p={p} minflux={minflux} maxflux={maxflux} binflux={binflux}")

    ep = 1 / (1.0 - p)
    bf2 = binflux * 2
    mbinflux = maxflux / binflux

    newvals = np.astype((p * rng.uniform(size=ntotal) ** ep + 0.5) / bf2,
                        np.int64)

    notvalid = (newvals < minflux) & (newvals >= mbinflux)
    while np.any(notvalid):
        nreplace = np.sum(notvalid)
        replacevals = np.astype((p * rng.uniform(size=nreplace) ** ep + 0.5) / bf2,
                                np.int64)
        newvals[notvalid] = replacevals
        notvalid = (newvals < minflux) & (newvals >= mbinflux)

    return newvals
