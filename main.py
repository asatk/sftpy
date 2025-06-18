"""
Model driver
"""

import numpy as np
from typing import Callable

from sftpy.charges import random_walk
from sftpy.cycle import cycle_modes
from sftpy.cycle import cyl_t
from sftpy.differential import diffr_modes
from sftpy.meridional import merid_modes
from sftpy.decay import decay

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
        dt=1,
        nstep=1000,
        seed=0x2025,
        nflux=2,
        nfluxmax=1000000,
        decay_t=1,
        inv_pol=1,
        source=1.0,
        phibins=360,
        thetabins=180,
        savelat=False,
        nstepfullres=1000000,
        as_specified=False,
        polarconverge=False,
        remhalf=False)

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

    # activity cycle strength mode
    mode_c = 1
    cycle = cycle_modes[mode_c]

    # differential rotation profile
    mode_d = 1
    diffr = diffr_modes[mode_d]

    # meridional flow behavior
    mode_m = 2
    merid = merid_modes[mode_m]


def loop():

    global params, phi, theta, flux, nflux, rng, cycle, diffr, merid

    time = 0.0
    dt = params.dt

    for i in range(params.nstep):

        print(i)

        time += dt

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
        if params.nstep - i < params.nstepfullres and params.as_specified == 0:
            pass
            """
            correction /= ff
            dt /= ff
            params.as_specified = True
            """


        nflux = decay(phi, theta, flux, nflux, params.dt, rng, params.decay_t)
        # safety catch -- NaN
        
        # calculate synoptic map
        synoptic, _, _ = np.histogram2d(
                phi, theta, bins=(params.phibins, params.thetabins),
                weights=np.fabs(flux))
        if params.savelat:
            hist_flux, _, _ = np.histogram2d(
                    phi, theta, bins=(params.phibins, params.thetabins),
                    weights=flux)
            np.save(hist_aflux, "aflux.npy")
            np.save(hist_flux, "flux.npy")

        # moves charges in random walk step size according to diffusion coeff
        synoptic = random_walk(phi, theta, flux, nflux, dt, rng, synoptic,
                               source=params.source)
        
        # safety catch -- NaN

        # apply differential rotation mode=4 o.w. use source diffr
        cycle(time)

        # apply meridional flow mode=4 o.w. use source merid
        cycle(time)

        # alternate applying merid and diffr steps

        # diffr dt/4 start point
        diffr(phi, theta, flux, nflux, dt/4)

        # merid dt/2 step 1
        merid(theta, nflux, dt/2)

        # diffr dt/2 midpoint
        diffr(phi, theta, flux, nflux, dt/2)

        # meriod dt/2 step 2
        merid(theta, nflux, dt/2)

        # diffr dt/4 endpoint
        diffr(phi, theta, flux, nflux, dt/4)

        # active region inflow towards regions of strong flux density

        # test for source collisions

        # track unsigned flux history

        # fragment concentrations

        # assimilate magnetogram data

        # add sources
        # add_sources()
        
        # forecasting

        # save timestep

        # print stuff

    # finish
        

if __name__ == "__main__":
    print("coming soon")
    init_sim()
    loop()
