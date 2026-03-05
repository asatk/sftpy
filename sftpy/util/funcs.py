import numba as nb
import numpy as np



@nb.jit(cache=True)
def consolidate(phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):
    """
    Removes spots with 0 flux. Returns number of remaining spots.
    """

    index = np.nonzero(flux[:nflux])[0]
    nnew = len(index)
    if nnew == nflux:
        return nflux

    phi[:nnew] = phi[index]
    theta[:nnew] = theta[index]
    flux[:nnew] = flux[index]

    return nnew



# replicates behavior of IDL SMOOTH
def smooth(arr: np.ndarray, width: int) -> np.ndarray:

    a = np.asarray(arr, dtype=np.float64, copy=True)

    if width % 2 == 0:
        width += 1

    lo = (width - 1) // 2
    hi = a.shape[0] - (width - 1) // 2
    kernel = np.full(width, 1. / width)
    a[lo:hi] = np.convolve(a, kernel, mode='valid')
    return a

def powerlaw_rv(ntotal: int, p: float, lo: float, hi: float, rng):

    # ks-test determined the IDL sampler and true power-law dists are not identical...
    # sampling via inverse method
    pp1 = p + 1
    urv = rng.uniform(size=ntotal)
    rv = (((hi ** pp1 - lo ** pp1) * urv) + lo ** pp1) ** (1 / pp1)

    return rv