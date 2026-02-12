import numba as nb
import numpy as np

def muarray(imsize: int):
    radius = imsize / 2.
    pos = (np.arange(imsize) + 0.5 - radius) / radius

    a = np.sqrt(np.add.outer(pos**2, pos**2))
    is_valid = a <= 1

    mu = np.full((imsize, imsize), -1, dtype=np.float64)
    mu[is_valid] = np.sin(np.acos(a[is_valid]))

    return mu

def tiltmatrix(tilt: float):
    """
    tilt: radians
    rot about y axis
    """
    rot = np.array([[ np.cos(tilt), 0.0, np.sin(tilt)],
                    [          0.0, 1.0,          0.0],
                    [-np.sin(tilt), 0.0, np.cos(tilt)]])
    return rot

def spinmatrix(angle: float):
    """
    angle: rad
    rot about z axis
    """
    rot = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle),  np.cos(angle), 0.0],
                    [          0.0,            0.0, 1.0]])
    return rot

def sumarr(arr: np.ndarray, ind: np.ndarray) -> np.ndarray:
    return np.sum(arr[ind])

# best guess at phithetaxyz
@nb.jit(cache=True)
def phithetaxyz(phi: np.ndarray, theta: np.ndarray, nflux: int) -> np.ndarray:
    sintheta = np.sin(theta[:nflux])
    x = sintheta * np.cos(phi[:nflux])
    y = sintheta * np.sin(phi[:nflux])
    z = np.cos(theta[:nflux])
    l = (x, y, z)
    r = np.stack(l, axis=1)
    return r
