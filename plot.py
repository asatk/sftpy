from matplotlib import pyplot as plt
import numpy as np

def plot_hist(h: np.ndarray, phibins: int=360, thetabins: int=180):
    ind = np.nonzero(h)

    im = plt.imshow(h.T, origin='upper', cmap="gray")
    plt.xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    plt.xlabel(r"Azimuth $\phi$")
    plt.yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    plt.ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.6, cmap="gray")
    plt.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")

    plt.show()


def plot_syn(phi: np.ndarray, theta: np.ndarray, flux: np.ndarray, nflux: int,
             phibins: int=360, thetabins: int=180, name: str=None):
    h, xe, ye = np.histogram2d(
            phi[:nflux], theta[:nflux], weights=flux[:nflux],
            bins=(phibins, thetabins), range=((0, 2*np.pi), (0, np.pi)))


    ind = np.nonzero(h)
    print("[plot] ---- LOC")
    print(ind)
    print("[plot] ---- FLUX[LOC]")
    print(h[ind])

    im = plt.imshow(h.T, origin='upper', cmap="gray", vmin=-3.0, vmax=3.0)
    plt.xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    plt.xlabel(r"Azimuth $\phi$")
    plt.yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    plt.ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.6, cmap="gray")
    plt.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")
    plt.title(name)

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_lat(phi: np.ndarray, theta: np.ndarray, flux: np.ndarray, nflux: int,
             phibins: int=360, thetabins: int=180):
    
    h_aflux, xe, ye = np.histogram2d(phi, theta, bins=(phibins, thetabins),
                                     weights=np.fabs(flux))
    h_flux, _, _ = np.histogram2d(
            phi, theta, bins=(params.phibins, params.thetabins),
            weights=flux)
    lat_aflux = np.sum(h_aflux, axis=0)
    lat_flux = np.sum(h_flux, axis=0)
