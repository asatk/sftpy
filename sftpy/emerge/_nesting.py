import cv2 as cv
import numpy as np

from sftpy import rng

from ..component import Component
from ..viz import plot_syn

class PlageNests(Component):

    prefix = "[plage-nest]"

    def __init__(self,
                 phibins: int,
                 thetabins: int,
                 binflux: float,
                 avefluxd: float,
                 thr: float,
                 nest_lat_lim: float,
                 loglvl: int=0):
        super().__init__(loglvl)

        self._phibins = phibins
        self._thetabins = thetabins
        self._binflux = binflux
        self._avefluxd = avefluxd
        self._active_thr = 2.5 * self._avefluxd * 1.47562 / 2 / binflux
        self._thr = thr
        self._nest_lat_lim = nest_lat_lim

    def identify_plages(self,
                        phi: np.ndarray,
                        theta: np.ndarray,
                        flux: np.ndarray,
                        nflux: int):

        # 2D histogram of unsigned flux in a sine latitude vs. longitude grid
        aflux = np.abs(flux[:nflux])
        costheta = np.cos(theta[:nflux])
        flux_hist, _, _ = np.histogram2d(
            phi[:nflux], costheta, weights=aflux,
            bins=(self._phibins, self._thetabins),
            range=((0, 2 * np.pi), (-1., 1.)))

        # threshold flux to include locations of plages -- binary map
        thr_gauss = self._thr / self._binflux * 1.4752
        synoptic_thr = np.astype(flux_hist > thr_gauss, np.float64)

        # TODO IDL -- compare smooth+dilation ops
        # looks good from debug plots in BMRSchrijver

        # smooth slightly and require at least 6 neighbors to be part of plage
        kernel = np.ones((3, 3), dtype=np.float64)
        synoptic_sm = cv.filter2D(synoptic_thr, -1, kernel / 9)

        # dilate to add an extra ring of pixels to plage
        synoptic_thr2 = np.asarray(synoptic_sm >= 6 / 9, dtype=np.uint8)
        synoptic_dil = cv.dilate(synoptic_thr2, kernel)

        # nesting latitude limit in pixels of a sine latitude grid
        sinlat_lim_px = np.int64((1 - np.sin(self._nest_lat_lim * np.pi / 180)) * self._thetabins / 2)

        # Sin latitude in pixels
        xx = np.zeros(self._thetabins, dtype=np.byte)
        xx[sinlat_lim_px:-sinlat_lim_px] = 1

        # Longitude in pixels
        yy = np.ones(self._phibins, dtype=np.byte)

        # Limit nesting w/in existing pages to latitudes btwn +/- nest_lat_lim
        mask = np.outer(yy, xx)

        # Identifies pixels with plages available for nesting (2D map)
        is_plage = synoptic_dil * mask

        return is_plage



    def place_active_regions(self,
                             phi: np.ndarray,
                             theta: np.ndarray,
                             flux: np.ndarray,
                             nflux: int,
                             newphi: np.ndarray,
                             newtheta: np.ndarray,
                             newflux: np.ndarray):
        # nesting
        # ~40% of activate regions emerge inside existing regions.
        # applied to all regions larger than 2.5 sq deg (factor 2 for 2 pol)
        # 1.4752 is flux to G
        is_active = np.nonzero(newflux >= self._active_thr)[0]
        nactive = len(is_active)
        if nactive > 0:
            # pick nest regions from set of sufficiently large regions
            will_nest = rng.uniform(size=nactive) < 0.4
            nnest = np.sum(will_nest)

            self.log(3, f"NEST nactive = {nactive}")

            # pick new location inside plage regions but not at polar caps
            # limits emergence to lat +/- deg
            if nnest > 0:
                is_nesting = is_active[will_nest]
                self.log(3, f"NEST nnest = {nnest}")

                # NOTE: nesting plages identified in IDL model much earlier
                # than immediately after sampling new spots. before flows,
                # fragmentation, and collisions.

                is_plage = self.identify_plages(phi, theta, flux, nflux)
                is_plage_px = np.array(np.nonzero(is_plage))
                nplage = is_plage_px.shape[1]

                if self._loglvl >= 3:
                    plot_syn(phi, theta, flux, nflux, show=True)

                self.plot(3, "imshow", is_plage.T)
                self.pshow(3)

                self.log(3, f"NEST nplage = {nplage}")

                if nplage > 0:
                    nreplace = min(nnest, nplage)
                    self.log(3, f"NEST nreplace = {nreplace}")
                    ind_chosen = rng.choice(nplage, replace=False, size=nreplace)
                    grid_chosen = is_plage_px[:,ind_chosen]

                    # TODO check the  +1 on these
                    nest_newphi = grid_chosen[0] / self._phibins * 2 * np.pi
                    nest_newtheta = np.arccos(grid_chosen[1] * 2 / self._thetabins - 1)
                    newphi[is_nesting[:nreplace]] = nest_newphi
                    newtheta[is_nesting[:nreplace]] = nest_newtheta

                    # self.log(3, f"NEST point: {point}")
                    # self.log(3, f"NEST lat: {lat}")
                    self.log(3, f"NEST newphi: {nest_newphi}")
                    self.log(3, f"NEST newlat: {nest_newtheta}")

                else:
                    self.log(3, "NEST no plage regions")
            else:
                self.log(3, "NEST no nesting regions")
        else:
            self.log(3, "NEST no new active regions")
