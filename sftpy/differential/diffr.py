from matplotlib import pyplot as plt
import numpy as np

dif_mult = 1.0
dif_thr = 0.0

def diffr_1(phi: np.ndarray,
            theta: np.ndarray,
            flux: np.ndarray,
            nflux: int,
            dt: float):

    if np.fabs(dif_mult) < 1.e-5:
        return

    mdiff = dif_mult

    # deg / day -> rad / step
    scale = dt * np.pi / 180 / 86400
    scale_mdiff = scale * mdiff

    a = (14.255984 - 14.18) * scale_mdiff
    b = -2.00 * scale_mdiff
    c = -2.09 * scale_mdiff

    sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
    phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a


def diffr_2(phi: np.ndarray,
            theta: np.ndarray,
            flux: np.ndarray,
            nflux: int,
            dt: float):

    if np.fabs(dif_mult) < 1.e-5:
        return

    mdiff = dif_mult
    # deg / day -> rad / step
    scale = dt * np.pi / 180 / 86400
    scale_mdiff = scale * mdiff

    a = (14.255984 - 14.18 + 0.2) * scale_mdiff
    
    b = -2.00 * scale_mdiff
    c = -2.09 * scale_mdiff

    sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
    phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a


def diffr_3(phi: np.ndarray,
            theta: np.ndarray,
            flux: np.ndarray,
            nflux: int,
            dt: float):

    if np.fabs(dif_mult) < 1.e-5:
        return

    mdiff = dif_mult
    # deg / day -> rad / step
    scale = dt * np.pi / 180 / 86400
    scale_mdiff = scale * mdiff

    a = (14.255984 - 14.18 + 0.2) * scale_mdiff
    
    b = -2.00 * scale_mdiff
    c = -2.09 * scale_mdiff

    sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
    phi[:nflux] += (b * sinlat2 + c * sinlat2**2) * \
        (0.8 * np.exp(-np.fabs(theta[:nflux]*180/np.pi - 90) / 80) + 0.2) + a



def diffr_4(phi: np.ndarray,
            theta: np.ndarray,
            flux: np.ndarray,
            nflux: int,
            dt: float,
            showmap: bool=False):

    if np.fabs(dif_mult) < 1.e-5:
        return
    
    mdiff = 1.0     # magnitude of differential rotation
    diff = np.fabs(dif_mult)    # magnitude of the relative flow
    sdiff = np.sign(dif_mult)
    
    # deg / day -> rad / step
    scale = dt * np.pi / 180 / 86400
    scale_mdiff = scale * mdiff

    a = (14.255984 - 14.18) * scale_mdiff

    b = -2.00 * scale_mdiff
    c = -2.09 * scale_mdiff

    sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2
    phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a

    # what if flux was limited to nflux elemetns instead of whole array

    # fast flux and slow flux
    if sdiff > 0:
        fflux = (flux >  dif_thr) & (theta >  np.pi/6) & (theta <   np.pi/2) | \
                (flux < -dif_thr) & (theta >= np.pi/2) & (theta < np.pi*5/6)
        sflux = (flux >  dif_thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                (flux < -dif_thr) & (theta >  np.pi/6) & (theta <   np.pi/2)
    else:
        fflux = (flux < -dif_thr) & (theta > np.pi*5/6) & (theta <  np.pi/2) | \
                (flux >  dif_thr) & (theta >=  np.pi/2) & (theta < np.pi*5/6)
        sflux = (flux < -dif_thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                (flux >  dif_thr) & (theta >  np.pi/6) & (theta < np.pi/2)

    scale2 = 10.0 / 7e8 * dt
    scale2_diff = scale2 * diff

    phi[fflux] += scale2_diff
    phi[sflux] -= scale2_diff
    
    if showmap:
        synmap(phi, theta, flux, fflux, sflux, nflux)


def synmap(phi: np.ndarray,
           theta: np.ndarray,
           flux: np.ndarray,
           fflux: np.ndarray,
           sflux: np.ndarray,
           nflux: int,
           phibins: int=360,
           thetabins: int=180):

    ssflux = np.copy(flux)
    ssflux[np.fabs(ssflux) < 50] = 0

    # TODO synoptic map prescription in schrijver + np.hist2d
    hist, _, _ = np.hist2d(phi, theta, bins=(phibins, thetabins),
                     weights=ssflux)
    hist = np.clip(hist, a_min=-50, a_max=50)
    h_min = np.min(hist)
    h_max = np.max(hist)

    # bytscl replacement
    hist_scale = np.asarray(((hist - h_min) / (h_max - h_min)) * 255, np.int64)
    # congrid replacement - 3x nearest-neighbor resampling of image
    hist_zoom = scipy.ndimage.zoom(hist_scale, 3, order=0)

    xf = phi[fflux] * 3 * 180 / np.pi
    yf = 3 * (np.sin(np.pi / 2 - theta[fflux]) * 90 + 89)
    xs = phi[sflux] * 3 * 180 / np.pi
    ys = 3 * (np.sin(np.pi / 2 - theta[sflux]) * 90 + 89)

    plt.imshow(hist_zoom / 4 + 96)
    plt.scatter(xf, yf, marker="+", color="black")
    plt.scatter(xs, ys, marker="-", color="red")
    plt.show()
