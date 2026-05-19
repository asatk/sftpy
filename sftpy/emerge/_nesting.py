import cv2 as cv
import numpy as np

def identify_nesting_plages(phi: np.ndarray,
                            theta: np.ndarray,
                            flux: np.ndarray,
                            nflux: int,
                            thr: float,
                            binflux: float,
                            phibins: int,
                            thetabins: int,
                            nest_lat_lim: float):

    # 2D histogram of unsigned flux in a sine latitude vs. longitude grid
    aflux = np.abs(flux[:nflux])
    costheta = np.cos(theta[:nflux])
    map, phi_edges, theta_edges = np.histogram2d(
        phi[:nflux], costheta, weights=aflux,
        bins=(phibins, thetabins),
        range=((0, 2 * np.pi), (-1., 1.)))

    # threshold flux to include locations of plages -- binary map
    synoptic_thr = np.astype(map > (thr / (binflux / 1.4752)), np.float64)

    # TODO IDL -- compare smooth+dilation ops
    # looks good from debug plots in BMRSchrijver

    # smooth slightly and require at least 6 neighbors to be part of plage
    kernel = np.ones((3, 3), dtype=np.float64)
    synoptic_sm = cv.filter2D(synoptic_thr, -1, kernel / 9)

    # dilate to add an extra ring of pixels to plage
    synoptic_thr2 = np.asarray(synoptic_sm > 5.9 / 9, dtype=np.uint8)
    synoptic_dil = cv.dilate(synoptic_thr2, kernel)

    # nesting latitude limit in pixels of a sine latitude grid
    sinlat_lim_px = np.int64((1 - np.sin(nest_lat_lim * np.pi / 180)) * thetabins / 2)

    # Sin latitude in pixels
    xx = np.zeros(thetabins, dtype=np.byte)
    xx[sinlat_lim_px:-sinlat_lim_px] = 1

    # Longitude in pixels
    yy = np.ones(phibins, dtype=np.byte)

    # Limit nesting w/in existing pages to latitudes btwn +/- nest_lat_lim
    mask = np.outer(yy, xx)

    # Identifies pixels with plages available for nesting (2D map)
    is_plage = synoptic_dil * mask

    return is_plage
