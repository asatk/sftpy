from matplotlib import pyplot as plt
import numpy as np
from ..general.congrid import congrid

def move_differential(phi: np.ndarray,
                      theta: np.ndarray,
                      flux: np.ndarray,
                      nflux: int,
                      dt: float,
                      differential: float,
                      mode: int=1,
                      thr: float=0.0
                      showmap: bool=False):
    
    if np.fabs(differential) < 1.e-5:
        return

    if mode < 1 or mode > 4:
        print('Stop: differential-rotation option not (yet) avalable')
        return

    if mode == 4:
        mdiff = 1.0     # magnitude of differential rotation
        diff = np.fabs(differential)    # magnitude of the relative flow
        sdiff = np.sign(differential)
    else:
        mdiff = differential

    # deg / day -> rad / step
    scale = dt * np.pi / 180 / 86400
    scale_mdiff = scale * mdiff

    a = (14.255984 - 14.18) * scale_mdiff

    if mode == 2 or mode == 3:
        a += 0.2 * scale_mdiff
    
    b = -2.00 * scale_mdiff
    c = -2.09 * scale_mdiff

    sinlat2 = np.sin(np.pi / 2 - theta[:nflux])**2

    if mode == 3:
        phi[:nflux] += (b * sinlat2 + c * sinlat2**2) * \
            (0.8 * np.exp(-np.fabs(theta[:nflux]*180/np.pi - 90) / 80) + 0.2) + a
    else:
        phi[:nflux] += b * sinlat2 + c * sinlat2**2 + a

    if mode == 4:
        # what if flux was limited to nflux elemetns instead of whole array

        # fast flux and slow flux
        if sdiff > 0:
            fflux = (flux >  thr) & (theta >  np.pi/6) & (theta <   np.pi/2) | \
                    (flux < -thr) & (theta >= np.pi/2) & (theta < np.pi*5/6)
            sflux = (flux >  thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                    (flux < -thr) & (theta >  np.pi/6) & (theta <   np.pi/2)
        else:
            fflux = (flux < -thr) & (theta > np.pi*5/6) & (theta <  np.pi/2) | \
                    (flux >  thr) & (theta >=  np.pi/2) & (theta < np.pi*5/6)
            sflux = (flux < -thr) & (theta >= np.pi/2) & (theta < np.pi*5/6) | \
                    (flux >  thr) & (theta >  np.pi/6) & (theta < np.pi/2)

        scale2 = 10.0 / 7e8 * dt
        scale2_diff = scale2 * diff

        phi[fflux] += scale2_diff
        phi[sflux] -= scale2_diff
        
        if showmap:
            ssflux = np.copy(flux)
            ssflux[np.fabs(ssflux) < 50] = 0

            # TODO synoptic map prescription in schrijver + np.hist2d
            phibins = 360
            thetabins = 180
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
