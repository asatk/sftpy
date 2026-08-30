import numpy as np
from scipy.io import readsav

def sav2npy(infiles: list[str], outfile: str=None,
            save_map: bool=True, save_spots: bool=False,
            phibins: int=360, thetabins: int=180):

    if not save_map and not save_spots:
        print("Nothing was saved.")
        return None

    # number of IDL workspace saves given by number of .sav files in dir
    nsteps_saved = len(infiles)

    save_dict = {}

    if save_map:
        # allocate memory for each saved map
        maps = np.empty((nsteps_saved, phibins, thetabins), dtype=np.int64)
        save_dict = {**save_dict,
                     "maps": maps}

    if save_spots:
        # allocate memory for each saved step
        phis = np.array([], dtype=np.float64)
        thetas = np.array([], dtype=np.float64)
        fluxes = np.array([], dtype=np.int64)
        nfluxes = np.empty(nsteps_saved, dtype=np.int32)

    for i, fn in enumerate(infiles):

        # read .sav IDL data
        data = readsav(fn)

        # retrieve spot locations and fluxes
        phi = data["phis"]
        theta = data["thetas"]
        flux = data["fluxs"]
        nflux = data["nflux"]

        if save_map:
            # project data onto Carringto map
            sinlat = np.cos(theta)
            map, _, _ = np.histogram2d(
                phi, sinlat, weights=flux,
                bins=(phibins, thetabins),
                range=((0, 2 * np.pi), (-1, 1)))
            maps[i] = map

        if save_spots:
            phis = np.append(phis, phi)
            thetas = np.append(thetas, theta)
            fluxes = np.append(fluxes, flux)
            nfluxes[i] = nflux

    if save_spots:
        save_dict = {**save_dict,
                     "phis": phis,
                     "thetas": thetas,
                     "fluxes": fluxes,
                     "nfluxes": nfluxes}

    if outfile is not None:
        np.savez_compressed(outfile, **save_dict)

    return maps
