import cv2 as cv
import numpy as np
import numpy.typing as npt

phibins = 360
thetabins = 180
thresh1 = 40
thresh2 = 6.0
area_lim = 100

_zero = np.int64(0.0)

def _scale(a: npt.NDArray[np.int64]) -> npt.NDArray[np.uint8]:
    return np.astype(a / np.max(a) * 255, np.uint8)

def identify_spot(phi: npt.NDArray,
                  theta: npt.NDArray,
                  flux: npt.NDArray,
                  nflux: int):
    sintheta = np.sin(theta[:nflux])
    map = np.histogram2d(phi[:nflux],
                             sintheta,
                             weights=flux[:nflux],
                             bins=(phibins, thetabins))
    mask_pos = map > _zero
    mask_neg = map < _zero

    mask_thresh1 = np.astype(np.abs(map) > thresh1, np.float64)

    nkernel_s = 3
    kernel_smooth = np.ones((nkernel_s, nkernel_s), dtype=np.float64)
    mask_smooth = cv.filter2D(mask_thresh1, -1, kernel_smooth)
    mask_thresh2 = np.astype(mask_smooth >= thresh2, np.float64)

    nkernel_d = 7
    kernel_dilate = np.ones((nkernel_d, nkernel_s), dtype=np.float64)
    mask_dilate = cv.dilate(mask_thresh2, kernel_dilate)

    map_pos = _scale(mask_pos * mask_dilate)
    map_neg = _scale(mask_neg * mask_dilate)

    cont_pos, _ = cv.findContours(map_pos, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cont_neg, _ = cv.findContours(map_neg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cont_pos_list = []
    for cont in cont_pos:
        if cont.shape[0] <= 2:
            continue

        cont_ = np.roll(cont, -1, axis=0)
        area = np.abs(np.sum((cont_[:, :, 0] + cont[:, :, 0]) * \
                             (cont_[:, :, 1] - cont[:, :, 1]))) / 2

        if area < area_lim:
            continue

        cont_pos_list.append(cont)

    cont_neg_list = []
    for cont in cont_neg:
        if cont.shape[0] <= 2:
            continue

        cont_ = np.roll(cont, -1, axis=0)
        area = np.abs(np.sum((cont_[:, :, 0] + cont[:, :, 0]) * \
                             (cont_[:, :, 1] - cont[:, :, 1]))) / 2

        if area < area_lim:
            continue

        cont_neg_list.append(cont)

    return cont_pos_list, cont_neg_list

    # TODO simplify polygon? using that opencv2 algo?

def find_concentrations():
    ...
    # TODO ID which flux concenctrations lie within contour

