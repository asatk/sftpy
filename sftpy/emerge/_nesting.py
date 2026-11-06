import numpy as np

from ..imgproc import ImageProcessor
from ..regions.constants import AR_BFIELD
from ..sim import Component, Data, rng, Timeline
from ..sim.constants import BINFLUX

from .constants import NEST_FIELD_MIN, NEST_LAT_MAX




class PlageNests(Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 impro: ImageProcessor,
                 ar_bfield: float = AR_BFIELD,
                 nest_field_min: float = NEST_FIELD_MIN,
                 nest_lat_max: float = NEST_LAT_MAX):
        super().__init__(time, data)
        self._impro = impro
        self._ar_bfield = ar_bfield
        self._nest_ar_flux_min = 2.5 * ar_bfield * 1.47562 / 2 / BINFLUX
        self._nest_field_min = nest_field_min * 1.4752 / BINFLUX
        self.nest_lat_max = nest_lat_max



    def identify_plages(self):

        data = self._data
        phi = data.phi
        theta = data.theta
        flux = data.flux
        nflux = data.nflux

        hist = self._impro.smooth_and_dilate(
            phi, theta, flux, nflux,
            absolute=True, threshold=self._nest_field_min)
        nphi, ntheta = hist.shape

        # nesting latitude limit in pixels of a sine latitude grid
        sinlat_lim_px = np.int64((1 - np.sin(self.nest_lat_max * np.pi / 180)) * ntheta / 2)

        # Sin latitude in pixels

        xx = np.zeros(ntheta, dtype=np.byte)
        xx[sinlat_lim_px:-sinlat_lim_px] = 1

        # Longitude in pixels
        yy = np.ones(nphi, dtype=np.byte)

        # Limit nesting w/in existing pages to latitudes btwn +/- nest_lat_lim
        mask = np.outer(yy, xx)

        # Identifies pixels with plages available for nesting (2D map)
        is_plage = hist * mask

        return is_plage



    def place_active_regions(self, data: Data):

        flux = data.flux
        phi = data.phi
        theta = data.theta

        # nesting
        # ~40% of activate regions emerge inside existing regions.
        # applied to all regions larger than 2.5 sq deg (factor 2 for 2 pol)
        # 1.4752 is flux to G
        is_active = np.asarray(flux >= self._nest_ar_flux_min).nonzero()[0]
        nactive = len(is_active)
        if nactive > 0:
            # pick nest regions from set of sufficiently large regions
            will_nest = rng.uniform(size=nactive) < 0.4
            nnest = np.sum(will_nest)

            # pick new location inside plage regions but not at polar caps
            # limits emergence to lat +/- deg
            if nnest > 0:
                is_nesting = is_active[will_nest]

                # NOTE: nesting plages identified in IDL model much earlier
                # than immediately after sampling new spots. before flows,
                # fragmentation, and collisions.

                is_plage = self.identify_plages()
                is_plage_px = np.array(np.nonzero(is_plage))
                nplage = is_plage_px.shape[1]

                if nplage > 0:
                    nreplace = min(nnest, nplage)
                    ind_chosen = rng.choice(nplage, replace=False, size=nreplace)
                    grid_chosen = is_plage_px[:,ind_chosen]

                    nphi, ntheta = is_plage.shape
                    nest_newphi = grid_chosen[0] / nphi * 2 * np.pi
                    nest_newtheta = np.arccos(grid_chosen[1] * 2 / ntheta - 1)
                    phi[is_nesting[:nreplace]] = nest_newphi
                    theta[is_nesting[:nreplace]] = nest_newtheta
