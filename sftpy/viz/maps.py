from matplotlib import animation as anim
from matplotlib import pyplot as plt
import numpy as np

from sftpy import simrc as rc

phibins = rc["viz.phibins"]
thetabins = rc["viz.thetabins"]
dt = rc["general.dt"] * rc["general.savestep"]
ms = rc["viz.ms"]
flux_thresh = rc["viz.maps.thr"]

def plot_syn(phi: np.ndarray,
             theta: np.ndarray,
             flux: np.ndarray,
             nflux: int,
             phibins: int=phibins,
             thetabins: int=thetabins,
             name: str=None,
             flux_thresh: int=flux_thresh,
             show: bool=False):

    sinlat = np.cos(theta[:nflux])
    h, xe, ye = np.histogram2d(
            phi[:nflux], sinlat, weights=flux[:nflux],
            bins=(phibins, thetabins), range=((0, 2*np.pi), (-1, 1)))

    fig, ax = plt.subplots(figsize=(6,3))

    ind = np.nonzero(h)
    
    im = ax.imshow(h.T, origin='upper', cmap="gray", vmin=-flux_thresh,
                    vmax=flux_thresh)
    ax.set_title(name)
    ax.set_xlabel(r"Azimuth $\phi$")
    ax.set_xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_ylabel(r"Colatitude $\theta$")
    ax.set_yticks([1, 0, -1], labels=[r"$\pi$", r"$\pi/2$", "0"])
    cb = plt.colorbar(im, shrink=0.9, cmap="gray")

    fig.tight_layout()

    if show:
        plt.show()



def anim_syn(synoptic_all: np.ndarray,
             dt: float=dt,
             phibins: int=phibins,
             thetabins: int=thetabins,
             flux_thresh: int=flux_thresh,
             ms: int=ms,
             show: bool=False):
    """
    Animate the evolution of the flux concentrations on the surface of a star.
    Displays the concentrations as a binned two-dimensional histogram.
    """

    figa, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))

    smap = np.transpose(synoptic_all, axes=(0,2,1))

    im = ax.imshow(smap[0], cmap="gray", vmin=-flux_thresh, vmax=flux_thresh,
                   extent=(0,phibins,0,thetabins), origin="upper")
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


def anim_map_with_flux(maps: np.ndarray,
                       dt: float=dt,
                       phibins: int=phibins,
                       thetabins: int=thetabins,
                       flux_thresh: int=flux_thresh,
                       ms: int=ms,
                       show: bool=False):
    figa, (axmap, axflux) = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))


    # Carrington Map plot
    maps = np.transpose(maps, axes=(0, 2, 1))

    im = axmap.imshow(maps[0], cmap="gray", vmin=-flux_thresh, vmax=flux_thresh,
                   extent=(0, phibins, 0, thetabins), origin="upper")
    # title = axmap.set_title("Frame {0:5d} ({t * dt / 86400:.01f} d)")
    axmap.set_title("Carrington Map")
    axmap.set_xticks([0, phibins // 2, phibins - 1],
                  labels=["0", r"$\pi$", r"$2\pi$"])
    axmap.set_xlabel(r"Azimuth $\phi$")
    axmap.set_yticks([thetabins - 1, thetabins // 2, 0],
                  labels=[r"$\pi$", r"$\pi/2$", "0"])
    axmap.set_ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.7, cmap="gray", label=r"Flux ($10^{18}$ Mx)")

    # Absolute ("Net") Flux plot
    aflux = np.sum(np.abs(maps), axis=(1, 2))
    nframes = maps.shape[0]
    time = np.arange(nframes) * dt / 86400 / 365

    axflux.plot(time, aflux)
    axflux.set_xlabel("Time (yr)")
    axflux.set_xlim(0, time[-1])
    axflux.set_ylabel("Flux ($10^{18}$ MX)")
    axflux.set_ylim(0, None)
    axflux.set_title("Total Absolute Flux")

    # correction to text location
    corr = 0.1 * (time - time[-1] / 2) / time[-1]
    flux_max = axflux.get_ylim()[1]

    point_flux = axflux.scatter(time[0], aflux[0], c="C1", s=25)
    line_flux = axflux.plot(time[0], aflux[0], c="C1")[0]
    text_flux = axflux.text(time[0]/time[-1], aflux[0]/flux_max - 0.05, f" {time[0]:.01f} yr ",
                            ha="center", va="top",
                            transform=axflux.transAxes)

    figa.tight_layout()

    def _update_map(t):
        im.set(data=maps[t])

        t_flux = time[t]
        point_flux.set_offsets([[t_flux, aflux[t]]])
        line_flux.set_data(time[:t], aflux[:t])


        # if t_flux > 0.9 * time[-1]:
        #     text_flux.set_ha("right")
        # else:
        #     text_flux.set_ha("left")

        text_flux.set_position([t_flux/time[-1] - corr[t], aflux[t]/flux_max - 0.05])
        text_flux.set_text(f" {t_flux:.01f} yr ")
        return im, line_flux, text_flux, point_flux
        # title.set_text(f"Frame {t:5d} ({t * dt / 86400:.01f} d)")
        # return im, title

    ani = anim.FuncAnimation(fig=figa, func=_update_map, frames=nframes,
                             interval=ms, blit=True)

    ani.save(filename="maps_flux.gif", writer="pillow")

    if show:
        plt.show()

