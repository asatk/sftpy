import numpy as np

def synoptic_map(phi: np.ndarray,
                 theta: np.ndarray,
                 vals: np.ndarray,
                 nvals: int,
                 phibins: int=360,
                 thetabins: int=180):
    hist, _, _ = np.histogram2d(
            phi[:nvals], theta[:nvals], weights=vals[:nvals],
            bins=(phibins, thetabins),
            range=((0, 2*np.pi), (0, np.pi)))
    return hist
