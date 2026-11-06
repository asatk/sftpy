import cv2 as cv
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
import numpy as np
import numpy.typing as npt
import re

phibins = 360
thetabins = 180
thresh1 = 40
thresh2 = 6.0
area_lim = 50

_zero = np.int64(0.0)

def _scale(a: npt.NDArray[np.int64]) -> npt.NDArray[np.uint8]:
    return np.astype(a / np.max(a) * 255, np.uint8)

# TODO temporary, will use a call to the track.spotid code in the future
def identify_spot(map: npt.NDArray):

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




p_num = re.compile(r"(\d+)")
buf_num = []
inc = 1
vlim = 100
norm = Normalize(vmin=-vlim, vmax=vlim, clip=True)



# maps = np.load("./runs/aug28five/maps.npz")["maps"]

import glob
flist = glob.glob("./runs/nov05run01/*.npy")
maps = np.array([np.load(f) for f in sorted(flist)])

iframe = 0
nframe = maps.shape[0]

map_i = maps[iframe]

fig, ax = plt.subplots()
im = ax.imshow(map_i.T, origin="lower", cmap="gray", norm=norm)
ax.set_xticks([0, phibins//2, phibins-1], labels=["0", r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"Azimuth $\phi$")
ax.set_yticks([thetabins-1, thetabins//2, 0], labels=[r"$\pi$", r"$\pi/2$", "0"])
ax.set_ylabel(r"Colatitude $\theta$")
title = ax.set_title(f"Magnetic Flux Map (Frame {iframe})")
text_skip = ax.annotate(f"increment: {inc}", (2, 2), xycoords="figure points",)
cb = fig.colorbar(im, ax=ax, shrink=0.6, label=r"Flux ($10^{18}$ Mx)")
# ax.scatter(ind[0], ind[1], marker="o", facecolors='none', color="orange")



all_contours = {}
toggle_contours = False
queue_contours = []

def on_press(event):

    global buf_num
    global iframe
    global im
    global inc
    global title
    global vlim

    global toggle_contours



    if event.key == "right":
        iframe = (iframe + inc) % nframe

    elif event.key == "left":
        iframe = (iframe - inc) % nframe

    elif p_num.match(event.key):
        buf_num.append(event.key)
        text_skip.set_text(f"increment: {''.join(buf_num)}*")

    elif event.key == "enter":
        if len(buf_num) > 0:
            inc = int("".join(buf_num))
        else:
            inc = 1

        text_skip.set_text(f"increment: {''.join(buf_num)}")
        buf_num = []

    elif event.key == "escape":
        text_skip.set_visible(not text_skip.get_visible())

    elif event.key == "up":
        vlim += inc
        norm = colors.Normalize(vmin=-vlim, vmax=vlim)
        im.set(norm=norm)

    elif event.key == "down":
        vlim -= inc
        norm = colors.Normalize(vmin=-vlim, vmax=vlim)
        im.set(norm=norm)

    elif event.key == "c":

        toggle_contours = not toggle_contours

    else:
        return

    map_i = maps[iframe]
    im.set(data=map_i.T)
    title.set_text(f"Magnetic Flux Map (Frame {iframe})")



    if len(ax.collections) == 2:
        for c in ax.collections:
            c.remove()

    if toggle_contours:

        if all_contours.get(iframe, None) is None:

            map_i = maps[iframe]
            contours_pos_list, contours_neg_list = identify_spot(map_i)

            patches_pos = []
            for i in range(0, len(contours_pos_list)):
                cont = contours_pos_list[i][:, 0, ::-1].copy()
                cont[:, 1] = cont[:, 1]
                poly = Polygon(cont, closed=True)
                patches_pos.append(poly)

            patches_neg = []
            for i in range(0, len(contours_neg_list)):
                cont = contours_neg_list[i][:, 0, ::-1].copy()
                cont[:, 1] = cont[:, 1]
                poly = Polygon(cont, closed=True)
                patches_neg.append(poly)

            blob_alpha = 0.5
            ppos = PatchCollection(patches_pos, color="green", alpha=blob_alpha,
                                   edgecolor="black")
            pneg = PatchCollection(patches_neg, color="orange",
                                   alpha=blob_alpha)

            if len(all_contours) >= 100:
                i_remove = queue_contours.pop(0)
                all_contours.pop(i_remove)

            all_contours[iframe] = [ppos, pneg]
            queue_contours.append(iframe)

        if all_contours.get(iframe, None):
            ppos = all_contours[iframe][0]
            pneg = all_contours[iframe][1]
            ax.add_collection(ppos)
            ax.add_collection(pneg)

    fig.canvas.draw()

fig.canvas.mpl_connect("key_press_event", on_press)

plt.show()