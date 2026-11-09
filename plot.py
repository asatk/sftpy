import numpy as np

from sftpy.viz import anim_map_with_flux, map_longavg, plot_aflux

if __name__ == "__main__":
    # fname = "maps.npy"
    # data = np.load(fname)
    # nflux = data.shape[0]

    import glob

    flist = glob.glob("./runs/nov07fivecycles/*.npy")
    data = np.array([np.load(f) for f in sorted(flist)])

    dt = 86400 * 1 * 16

    plot_aflux(data, dt, show=True)

    map_longavg(data, dt, flux_thresh=10)

    anim_map_with_flux(data, dt, show=True, ms=100, format="mkv")


