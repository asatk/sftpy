
from re import sub as re_sub

import cv2 as cv
import numpy as np

from ..constants import rng

def move_charges(phi: np.ndarray,
                theta: np.ndarray,
                flux,
                binflux,
                nflux: int,
                dt,
                diffusion: int,
                dependence,
                synoptic: np.ndarray,
                source,
                difftest,
                savelat: bool=False):

    # step size array
    step = np.ones(nflux, dtype=np.float64)

    bintheta = 360
    binphi = 180

    # threshold in gauss for map of absolute flux density
    thr = 40.0
    # synoptic bitmap
    synoptic = np.zeros((360, 180), dtype=np.float64)
    
    x = np.clip(np.asarray(
        np.mod((phi[:nflux] + 2*np.pi), 2*np.pi) * 180 / np.pi,
        np.int64), a_min=0, a_max=359)
    y = np.clip(np.asarray(
        (np.sin(theta[:nflux] + np.pi/2) + 1) * 90,
        np.int64), a_min=0, a_max=179)
    c = x + y * 360 # indexing 1d arr
    aflux = np.fabs(flux)

    synoptic = np.reshape(synoptic, 360 * 180)

    # sumarr, nflux, y, aflux, s
    # sumarr syntax: num elements, indicies, array, destination

    # this is basically sumarr
    # sumarr,nflux,c,aflux,synoptic
    # former, much slower statement
    #for i=0l,nflux-1l do synoptic(x(i),y(i))=synoptic(x(i),y(i))+abs(flux(i))
    
    # TODO IDL -- compare hist2d and sumarr stuff
    # map of absolute fluxes

    # longitudinally summed absolute fluxes
    if savelat:
        hist_flux = np.hist2d(phi, theta, bins=bintheta, weights=flux)
        np.save(hist_aflux, "aflux.npy")
        np.save(hist_flux, "flux.npy")


    if source < 0:
        # isn't np.fabs redundant?
        synoptic_abs = np.fabs(synoptic)
        temp = 10 / (binflux / 1.4752)
        # where is difftest used
        difftest = np.sum(synoptic_abs[synoptic_abs > temp]) * dt

    # threshold out flux to only include plages
    synoptic_thr = synoptic[synoptic > thr / (binflux / 1.4752)]

    # TODO IDL -- compare smooth+dilation ops
    # scipy.ndimage.filters.uniform_filter(data,size=3)
    # smooth slightly and require at least 6 neighbors to be part of plage
    #np.convolve(synoptic_thr, np.ones(3) / 3, mode="same")
    synoptic_sm = cv.filter2D(synoptic, -1,
                              np.ones((3,3), dtype=np.float64)/9)
    # dilate to add an extra ring of pixels to plage
    #synoptic = cv.dilate(synoptic_smooth[synoptic_smooth > 5.9/9],
    #    np.ones((3,3), dtype=np.int64)))
    synoptic_dil = cv.dilate(np.asarray(synoptic_sm > 5.9/9, dtype=np.int64), -1,
                             np.ones((3,3), dtype=np.float64))

    # if required step size adjusted using ratio of diffusion coefficients
    # from Schrijver&Martin 90; not done in conjunction w/ flux-dens dep
    if dependence == 1:
        ind1 = synoptic_dil[x, y] > 0
        step[ind1] = 110./250
    
    # evaluate the actual stepping distance
    step = np.sqrt(4 * diffusion * step * dt) / 7.e5

    # if flux-dependent steps are required, apply correction to concentrations
    # contained in a plage. See Schrijver+ 96 and PhD thesis Hagenaar p119 f7.4.
    if dependence == 2:
        step *= (240. / 140.) * np.exp(-np.fabs(flux[:nflux]) * binflux/35.)
    
    # move the sources given the above step size:
    # 1) rotate sphere so point is on the pole
    # 2) tilt over step in theta
    # 3) rotate over random phi angle
    # 4) apply inverse of 1
    # 5) step 1 approximated by letting concentrations move on a plane tanget
    # to the sphere at the pole, neglecting curvature

    # random direction to step in
    rphi = rng.uniform(high=2*np.pi, size=nflux)
    cosrphi = np.cos(rphi)
    sinrphi = np.sin(rphi)
    
    cosphi = np.cos(phi[:nflux])
    sinphi = np.sin(phi[:nflux])

    costheta = np.cos(theta[:nflux])
    sintheta = np.sin(theta[:nflux])

    crct = cosrphi * costheta
    x = cosphi * sintheta + step * (crct * cosphi - sinrphi * sinphi)
    y = sinphi * sintheta + step * (crct * sinphi + sinrphi * cosphi)
    z = costheta - step * cosrphi * sintheta

    phi[:nflux] = np.mod(np.arctan2(y, x) + 2*np.pi, 2*np.pi)
    theta[:nflux] = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))

    return synoptic_dil
