import numpy as np

from sftpy.viz import plot_syn, plot_lat, plot_hist, anim_syn, plot_aflux, \
    anim_map_with_flux

if __name__ == "__main__":
    # fname = "maps.npy"
    # data = np.load(fname)
    # nflux = data.shape[0]

    import glob

    flist = glob.glob("./runs/nov05run01/*.npy")
    data = np.array([np.load(f) for f in sorted(flist)])

    dt = 86400 * 1 * 8

    anim_map_with_flux(data, dt, show=True, ms=100, format="mkv")