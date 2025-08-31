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
             flux_thresh: int=100.0, show: bool=False):

    h, xe, ye = np.histogram2d(
            phi[:nflux], theta[:nflux], weights=flux[:nflux],
            bins=(phibins, thetabins), range=((0, 2*np.pi), (0, np.pi)))

    fig, ax = plt.subplots(figsize=(6,3))

    ind = np.nonzero(h)
    
    im = ax.imshow(h.T, origin='upper', cmap="gray", vmin=-flux_thresh,
                    vmax=flux_thresh)
    #plt.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")
    ax.set_title(name)
    ax.set_xlabel(r"Azimuth $\phi$")
    ax.set_xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_ylabel(r"Colatitude $\theta$")
    ax.set_yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    cb = plt.colorbar(im, shrink=0.9, cmap="gray")

    fig.tight_layout()

    if show:
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


def anim_syn(synoptic_all: np.ndarray, dt: float, phibins: int=360,
             thetabins: int=180, flux_thresh: int=100.0, ms: int=100,
             show: bool=False):
    """
    Animate the evolution of the flux concentrations on the surface of a star.
    Displays the concentrations as a binned two-dimensional histogram.
    """

    figa, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))

    smap = np.transpose(synoptic_all, axes=(0,2,1))

    im = ax.imshow(smap[0], cmap="gray", vmin=-flux_thresh, vmax=flux_thresh,
                   extent=(0,phibins,0,thetabins))
    title = ax.set_title("Frame {0:5d} ({t * dt / 86400:.01f} d)")
    ax.set_xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"Azimuth $\phi$")
    ax.set_yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    ax.set_ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.9, cmap="gray", label=r"Flux ($10^{18}$ Mx)")
    figa.tight_layout()

    def _update(t):
        im.set(data=smap[t])
        title.set_text(f"Frame {t:5d} ({t * dt / 86400:.01f} d)")
        return im, title

    ani = anim.FuncAnimation(fig=figa, func=_update, frames=len(smap),
                             interval=ms, blit=True)

    ani.save(filename="maps.gif", writer="pillow")

    if show:
        plt.show()


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


# TODO separate plot script from plotting fns. move fns to lib.
if __name__ == "__main__":
    fname = "maps.npy"
    data = np.load(fname)
    nflux = data.shape[0]

    dt = 21600 * 10

    plot_aflux(data, dt, show=True)
    anim_syn(data, dt, show=True)
