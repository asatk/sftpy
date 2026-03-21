import numpy as np

from sftpy.viz import plot_syn, plot_lat, plot_hist, anim_syn, plot_aflux, \
    anim_map_with_flux

if __name__ == "__main__":
    fname = "maps.npy"
    data = np.load(fname)
    nflux = data.shape[0]

    dt = 21600 * 4 * 25

    anim_map_with_flux(data, dt, show=True, ms=100, format="mkv")