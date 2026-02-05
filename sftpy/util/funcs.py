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