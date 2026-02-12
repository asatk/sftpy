import numba as nb
import numpy as np

@nb.jit(cache=True)
def consolidate(phi: np.ndarray,
                theta: np.ndarray,
                flux: np.ndarray,
                nflux: int):

    index = np.nonzero(flux[:nflux])[0]
    nnew = len(index)
    if nflux == nnew:
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