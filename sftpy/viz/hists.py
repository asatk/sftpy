from matplotlib import pyplot as plt
import numpy as np

from ..imgproc.constants import NPHI, NTHETA



def plot_hist(h: np.ndarray,
              nphi: int = NPHI,
              ntheta: int = NTHETA,
              show: bool=False):

    fig, ax = plt.subplot()

    # ind = np.nonzero(h)

    im = ax.imshow(h.T, origin='upper', cmap="gray")
    ax.xticks([0, nphi // 2, nphi - 1], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.xlabel(r"Azimuth $\phi$")
    ax.yticks([ntheta - 1, ntheta // 2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    ax.ylabel(r"Colatitude $\theta$")
    cb = ax.colorbar(im, shrink=0.6, cmap="gray")
    # ax.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")

    if show:
        plt.show()

    return fig, ax


def plot_lat(phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             nphi: int = NPHI,
             ntheta: int = NTHETA):
    
    h_aflux, xe, ye = np.histogram2d(phi, theta, bins=(nphi, ntheta),
                                     weights=np.fabs(flux))
    h_flux, _, _ = np.histogram2d(
            phi, theta, bins=(nphi, ntheta),
            weights=flux)
    lat_aflux = np.sum(h_aflux, axis=0)
    lat_flux = np.sum(h_flux, axis=0)


def plot_aflux(maps: np.ndarray,
               dt: float,
               fname: str | None = None,
               show: bool = False):
    """
    Plot the total absolute flux of the stellar surface.
    """

    fig, ax = plt.subplots()

    nframe = maps.shape[0]
    aflux = np.sum(np.abs(maps), axis=(1, 2))
    time = np.arange(nframe) * dt / 86400 / 365.25

    ax.plot(time, aflux)
    ax.set_xlabel("Time (yr)")
    ax.set_xlim(0, None)
    ax.set_ylabel("Flux ($10^{18}$ MX)")
    ax.set_ylim(0, None)
    ax.set_title("Total Absolute Flux")

    fig.tight_layout()

    if show:
        plt.show()

    if fname is not None:
        fig.savefig(fname=fname)

