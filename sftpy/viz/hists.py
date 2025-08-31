from matplotlib import animation as anim
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


def plot_lat(phi: np.ndarray, theta: np.ndarray, flux: np.ndarray, nflux: int,
             phibins: int=360, thetabins: int=180):
    
    h_aflux, xe, ye = np.histogram2d(phi, theta, bins=(phibins, thetabins),
                                     weights=np.fabs(flux))
    h_flux, _, _ = np.histogram2d(
            phi, theta, bins=(params.phibins, params.thetabins),
            weights=flux)
    lat_aflux = np.sum(h_aflux, axis=0)
    lat_flux = np.sum(h_flux, axis=0)


def plot_aflux(synoptic_all: np.ndarray, dt: float, show: bool=False):
    """
    Plot the total absolute flux of the stellar surface.
    """

    fig, ax = plt.subplots()

    nframe = synoptic_all.shape[0]
    aflux = np.sum(np.fabs(synoptic_all), axis=(1,2))
    time = np.arange(nframe) * dt / 86400 / 365

    ax.plot(time, aflux)
    ax.set_xlabel("Time (yr)")
    ax.set_xlim(0, None)
    ax.set_ylabel("Flux ($10^{18}$ MX)")
    ax.set_ylim(0, None)
    ax.set_title("Total Absolute Flux")

    fig.tight_layout()

    if show:
        plt.show()

