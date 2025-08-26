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


def plot_syn(phi: np.ndarray, theta: np.ndarray, flux: np.ndarray, nflux: int,
             phibins: int=360, thetabins: int=180, name: str=None,
             save: bool=False):
    h, xe, ye = np.histogram2d(
            phi[:nflux], theta[:nflux], weights=flux[:nflux],
            bins=(phibins, thetabins), range=((0, 2*np.pi), (0, np.pi)))


    ind = np.nonzero(h)
    """
    print("[plot] ---- LOC")
    print(ind)
    print("[plot] ---- FLUX[LOC]")
    print(h[ind])
    """
    
    im = plt.imshow(h.T, origin='upper', cmap="gray", vmin=-6.0, vmax=6.0)
    plt.xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    plt.xlabel(r"Azimuth $\phi$")
    plt.yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    plt.ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.6, cmap="gray")
    #plt.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")
    plt.title(name)

    if save:
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


def anim_syn(synoptic_all: np.ndarray, dt: float, phibins: int=360, thetabins: int=180):

    figa, ax = plt.subplots(nrows=1, ncols=1)

    im = ax.imshow(synoptic_all[0].T, cmap="gray", vmin=-100.0, vmax=100.0)
    ax.set_xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"Azimuth $\phi$")
    ax.set_yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    ax.set_ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.6, cmap="gray")
    figtitle = figa.suptitle("Frame 0 ({t * dt / 86400:.01f} d)")

    def _update(t):
        im = ax.imshow(synoptic_all[t].T, cmap="gray", vmin=-6.0, vmax=6.0)
        figtitle.set_text(f"Frame {t} ({t * dt / 86400:.01f} d)")
        return im, figtitle

    ms = 100
    ani = anim.FuncAnimation(fig=figa, func=_update, frames=len(synoptic_all),
                             interval=ms)
    ani.save(filename="maps.gif", writer="pillow")
    # plt.show()
