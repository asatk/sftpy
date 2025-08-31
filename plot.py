import numpy as np

from sftpy.viz import plot_syn, plot_lat, plot_hist, anim_syn, plot_aflux

if __name__ == "__main__":
    fname = "maps.npy"
    data = np.load(fname)
    nflux = data.shape[0]

    dt = 21600 * 10

    plot_aflux(data, dt, show=True)
    anim_syn(data, dt, show=True)
