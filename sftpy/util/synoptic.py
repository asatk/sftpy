import numpy as np
from sftpy import simrc as rc

phibins = rc["viz.phibins"]
thetabins = rc["viz.thetabins"]

def synoptic_map(phi: np.ndarray,
                 theta: np.ndarray,
                 vals: np.ndarray,
                 nvals: int,
                 phibins: int=phibins,
                 thetabins: int=thetabins):
    hist, _, _ = np.histogram2d(
            phi[:nvals], theta[:nvals], weights=vals[:nvals],
            bins=(phibins, thetabins),
            range=((0, 2*np.pi), (0, np.pi)))
    return hist
