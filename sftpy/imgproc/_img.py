import cv2 as cv
import numpy as np
import numpy.typing as npt

from ..sim import Component, Data, Timeline

from .constants import NPHI, NTHETA, SIZE_KERNEL_SMOOTH, SIZE_KERNEL_DILATE
from .constants import SMOOTH_THRESHOLD, THRESHOLD



class ImageProcessor(Component):

    def __init__(self,
                 time: Timeline,
                 data: Data,
                 nphi: int = NPHI,
                 ntheta: int = NTHETA,
                 nkernel_smooth: int = SIZE_KERNEL_SMOOTH,
                 nkernel_dilate: int = SIZE_KERNEL_DILATE):
        super().__init__(time, data)
        self._nphi = nphi
        self._ntheta = ntheta
        self._nkernel_smooth = nkernel_smooth
        self._nkernel_dilate = nkernel_dilate

        self._kernel_smooth: npt.NDArray = np.ones(
            (nkernel_smooth, nkernel_smooth), dtype=np.float64)
        self._kernel_dilate: npt.NDArray = np.ones(
            (nkernel_smooth, nkernel_smooth), dtype=np.float64)

    def bin_flux(self,
                 phi: npt.NDArray[np.float64],
                 theta: npt.NDArray[np.float64],
                 flux: npt.NDArray[np.int64],
                 nflux: int,
                 absolute: bool = False):
        x = phi[:nflux]
        y = np.cos(theta[:nflux])
        z = flux[:nflux]
        if absolute:
            z = np.abs(z)

        hist, phi_edges, theta_edges = np.histogram2d(
            x, y, weights=z,
            bins=(self._nphi, self._ntheta),
            range=((0, 2 * np.pi), (-1, 1)))
        return hist

    def smooth_and_dilate(self,
                          phi: npt.NDArray[np.float64],
                          theta: npt.NDArray[np.float64],
                          flux: npt.NDArray[np.int64],
                          nflux: int,
                          absolute: bool = False,
                          threshold: float = THRESHOLD,
                          smooth_threshold: float = SMOOTH_THRESHOLD):
        x = phi[:nflux]
        y = np.cos(theta[:nflux])
        z = flux[:nflux]
        if absolute:
            z = np.abs(z)

        hist, phi_edges, theta_edges = np.histogram2d(
            x, y, weights=z,
            bins=(self._nphi, self._ntheta),
            range=((0, 2 * np.pi), (-1, 1)))

        # threshold flux to include locations of plages -- binary map
        hist_thresh = np.asarray(hist > threshold, np.float64)

        # smooth slightly and require at least 6 neighbors to be part of plage
        hist_smooth = cv.filter2D(
            hist_thresh,
            -1,
            self._kernel_smooth / self._nkernel_smooth**2)

        smooth_threshold /= self._nkernel_smooth ** 2
        hist_smooth_thresh = np.asarray(
            hist_smooth >= smooth_threshold, dtype=np.uint8)

        # dilate to add an extra ring of pixels to plage
        hist_dil = cv.dilate(hist_smooth_thresh, self._kernel_dilate)

        return hist_dil
