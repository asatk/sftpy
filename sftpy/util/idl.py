import numpy as np
from scipy.io import readsav

def sav2npy(infiles: list[str], outfile: str=None, phibins: int=360, thetabins: int=180):

    nsteps_saved = len(infiles)

    # allocate memory for each saved map
    maps = np.empty((nsteps_saved, phibins, thetabins), dtype=np.int64)

    for i, fn in enumerate(infiles):
        # read .sav IDL data
        data = readsav(fn)

        # retrieve spot locations and fluxes
        phi = data["phis"]
        theta = data["thetas"]
        flux = data["fluxs"]

        sinlat = np.cos(theta)
        map, _, _ = np.histogram2d(
            phi, sinlat, weights=flux,
            bins=(phibins, thetabins),
            range=((0, 2 * np.pi), (-1, 1)))
        maps[i] = map

    if outfile is not None:
        np.savez_compressed(outfile, maps=maps)

    return maps
