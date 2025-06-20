from matplotlib import pyplot as plt
import numpy as np

def plot_hist(h: np.ndarray, phibins: int=360, thetabins: int=180):
    ind = np.nonzero(h)
    print(ind)
    print(h[ind])

    im = plt.imshow(h.T, origin='upper')
    plt.xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    plt.xlabel(r"Azimuth $\phi$")
    plt.yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    plt.ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.6)
    plt.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")

    plt.show()


def plot_syn(phi: np.ndarray, theta: np.ndarray, flux: np.ndarray, nflux: int,
         phibins: int=360, thetabins: int=180):
    h, xe, ye = np.histogram2d(
            phi[:nflux], theta[:nflux], weights=flux[:nflux],
            bins=(phibins, thetabins), range=((0, 2*np.pi), (0, np.pi)))


    print("----")
    print(phi[:nflux])
    print("----")
    print(theta[:nflux])
    print("----")
    print(flux[:nflux])
    print("----")

    ind = np.nonzero(h)
    print(ind)
    print(h[ind])

    """
    # im = plt.imshow(h, extent=[0, 2*np.pi, 0, np.pi])
    plt.xticks([0, np.pi, 2*np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    plt.yticks([np.pi, np.pi/2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    cb = plt.colorbar(im, shrink=0.6)
    # plt.scatter(phi[:nflux], theta[:nflux], marker="o", facecolors='none', color="orange")
    """

    im = plt.imshow(h.T, origin='upper')
    plt.xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
    plt.xlabel(r"Azimuth $\phi$")
    plt.yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
    plt.ylabel(r"Colatitude $\theta$")
    cb = plt.colorbar(im, shrink=0.6)
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
