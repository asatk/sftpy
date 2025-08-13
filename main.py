"""
Model driver
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Callable

from plot import plot_syn, plot_lat, plot_hist, anim_syn

from sftpy.ar import add_sources
from sftpy.charges import random_walk
from sftpy.collide import collide
from sftpy.cycle import cycle_modes
from sftpy.cycle import cyl_t
from sftpy.differential import diffr_modes
from sftpy.meridional import merid_modes
from sftpy.decay import decay
from sftpy.util import synoptic_map

class Params(dict):
    
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

# declare functions
cycle: Callable
diffr: Callable
merid: Callable

# declare data arrays
phi: np.ndarray
theta: np.ndarray
flux: np.ndarray
nflux: int      # number of flux concentrations
rng: np.random.Generator
params: Params


def init_sim() -> dict:

    global params, phi, theta, flux, nflux, rng, cycle, diffr, merid

    params = Params(
        dt=21600,   # 6 hrs
        nstep=100,
        savestep=1,
        seed=0x2025,
        nflux=2,
        nfluxmax=250000,
        decay_t=1,
        inv_pol=1,
        source=1.0,
        phibins=360,
        thetabins=180,
        savelat=False,
        nstepfullres=1000000,
        as_specified=True,
        polarconverge=False,
        mode_c=1,
        mode_d=1,
        mode_m=2,
        remhalf=False,
        correction=1.0,
        ff=1.0,
        outfile="maps.npy")

    rng = np.random.default_rng(seed=params.seed)

    phi = np.zeros(params.nfluxmax, dtype=np.float64)
    phi[0] = np.pi * 20 / 180
    phi[1] = np.pi * 20 / 180
    phi[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

    theta = np.zeros(params.nfluxmax, dtype=np.float64)
    theta[0] = np.pi * 107 / 180
    theta[1] = np.pi *  73 / 180
    theta[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

    flux = np.zeros(params.nfluxmax, dtype=np.int64)
    flux[0] = 3 * params.inv_pol
    flux[1] = -3 * params.inv_pol

    nflux = 2
    
    print("initial")
    plot_syn(phi, theta, flux, nflux, name="map-initial.png")

    # activity cycle strength mode
    cycle = cycle_modes[params.mode_c]

    # differential rotation profile
    diffr = diffr_modes[params.mode_d]

    # meridional flow behavior
    merid = merid_modes[params.mode_m]


def loop():

    global params, phi, theta, flux, nflux, rng, cycle, diffr, merid

    nstep = params.nstep
    dt = params.dt
    time = dt
    source = params.source
    as_specified = params.as_specified
    correction = params.correction
    savestep = params.savestep
    outfile = params.outfile

    kwargs_d = {}
    kwargs_m = {}

    # save synoptic maps at regular intervals
    synoptic_all = np.empty(((nstep - 1) // savestep + 1, 360, 180), dtype=np.int64)

    # save initial step
    synoptic_save = synoptic_map(phi, theta, flux, nflux)
    synoptic_all[0] = synoptic_save

    for i in range(1, nstep):

        if params.mode_d == 4:
            kwargs_d = dict(cycle=cycle,time=time)

        if params.mode_m == 4:
            kwargs_m = dict(cycle=cycle,time=time)


        print(f"[{i:8d}] -- time {time:.02e}")
        # plot_syn(phi, theta, flux, nflux, name=f"map{i:05d}.png")

        # polar converge
        # TODO do every half-cycle or just first half-cycle?
        if np.fmod(time + cyl_t, cyl_t) > cyl_t / 2 and not params.polarconverge and params.remhalf:
            ind = np.choice(nflux/2)
            netflux = np.sum(flux[ind])
            flux[:nflux/2] = flux[ind]
            phi[:nflux/2] = phi[ind]
            theta[:nflux/2] = theta[ind]
            nflux = len(ind)

            # TODO is this the weird flux normalization thing?
            # see L662 kit.pro (search 'Flux imbalance 0')
            flux[nflux/2:] = flux[nflux/2:] - netflux # ensure zero totalflux
            params.polarconverge = True


        # scale control params back
        if params.nstep - i < params.nstepfullres and as_specified == 0:
            correction /= params.ff
            dt /= params.ff
            as_specified = True

        print(f"[{i:8d}|decay]")
        nflux = decay(phi, theta, flux, nflux, params.dt, rng, params.decay_t)
        # safety catch -- NaN
        
        # calculate synoptic map
        synoptic = synoptic_map(phi, theta, np.fabs(flux), nflux)

        if params.savelat:
            hist_aflux = synoptic
            hist_flux = synoptic_map(phi, theta, flux, nflux)
            lat_aflux = np.sum(hist_aflux, axis=0)
            lat_flux = np.sum(hist_flux, axis=0)
            np.save(lat_aflux, "lat_aflux.npy")
            np.save(lat_flux, "lat_flux.npy")

        # moves charges in random walk step size according to diffusion coeff

        print(f"[{i:8d}|random_walk]")
        synoptic = random_walk(phi, theta, flux, nflux, dt, rng, synoptic,
                               source=source)
        # safety catch -- NaN

        # alternate applying merid and diffr steps

        # diffr dt/4 start point
        print(f"[{i:8d}|diffr] ---- dt/4, startpt")
        diffr(phi, theta, flux, nflux, dt/4, **kwargs_d)

        # merid dt/2 step 1
        print(f"[{i:8d}|merid] ---- dt/2, step1")
        merid(theta, nflux, dt/2, **kwargs_m)

        # diffr dt/2 midpoint
        print(f"[{i:8d}|diffr] ---- dt/2, midpt")
        diffr(phi, theta, flux, nflux, dt/2, **kwargs_d)

        # meriod dt/2 step 2
        print(f"[{i:8d}|merid] ---- dt/2, step2")
        merid(theta, nflux, dt/2, **kwargs_m)

        # diffr dt/4 endpoint
        print(f"[{i:8d}|diffr] ---- dt/4, endpt")
        diffr(phi, theta, flux, nflux, dt/4, **kwargs_d)

        # active region inflow towards regions of strong flux density

        # test for source collisions
        mode_col = 2
        nflux = collide(phi, theta, flux, nflux, dt, rng, mode_col, correction, trackcancel=True)

        # track unsigned flux history

        # fragment concentrations

        # assimilate magnetogram data

        # add sources
        print(f"[{i:8d}|cycle]")
        source_str, latsource = cycle(time)
        source_str *= params.inv_pol * 2 - 1
        # print


        print(f"[{i:8d}|add_src]")
        nflux = add_sources(phi, theta, flux, nflux, dt, rng, source_str,
                            latsource, synoptic)
        
        # forecasting

        # save timestep

        # print stuff

        if i % savestep == 0:
            synoptic_save = synoptic_map(phi, theta, flux, nflux)
            synoptic_all[i//savestep] = synoptic_save

        time += dt
        print()

    # finish

    plot_syn(phi, theta, flux, nflux, name=f"map-final.png")

    return synoptic_all

        

if __name__ == "__main__":
    outfile = "maps.npy"

    init_sim()
    synoptic_all = loop()
    np.save(outfile, synoptic_all)
    anim_syn(synoptic_all)
    plt.show()
