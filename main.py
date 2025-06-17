"""
Model driver
"""

import numpy as np
from typing import Callable


from .charges import charges
from .collide import collide
from .cycle import cycle
from .differential import differential
from .emergence import add_sources
from .fielddecay import fielddecay
from .meridional import merid_modes

merid: Callable
diffr: Callable

def init_sim() -> dict:

    theta = np.zeros(nfluxmax, dtype=np.float64)
    theta[0] = np.pi * 107 / 180
    theta[1] = np.pi *  73 / 180
    theta[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

    phi = np.zeros(nfluxmax, dtype=np.float64)
    phi[0] = np.pi * 20 / 180
    phi[1] = np.pi * 20 / 180
    phi[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

    flux = np.zeros(nfluxmax, dtype=np.int64)
    flux[0] = 3 * inv_pol
    flux[1] = -3 * inv_pol

    rng = np.random.default_rng(seed=seed)



    params = dict(
        dt=1,
        nstep=1000,
        seed=0x2025,
        nflux=2,
        decay_t=1)

    return theta, phi, flux, rng, params 


def loop(theta, phi, flux, params):

    for i in range(params.nstep):
    
        # polar converge
        # scale control params back


        nflux = fielddecay(theta, phi, flux, rng, nflux, params.dt, params.decay_t)
        # safety catch -- NaN
        
        synoptic = charges(theta, phi, flux, rng, nflux,)
        # safety catch -- NaN

        # apply differential rotation mode=4 o.w. use source diffr
        cycle()

        # apply meridional flow mode=4 o.w. use source merid
        cycle()

        # alternate applying merid and diffr steps

        # diffr dt/4 start point
        differential(phi, theta, flux, nflux, dt/4, params.diffr,
                     mode=params.mode_d, thr=params.thr)

        # merid dt/2 step 1
        merid(theta, nflux, dt/2, params.merid)

        # diffr dt/2 midpoint
        differential(phi, theta, flux, nflux, dt/2, params.diffr,
                     mode=params.mode_d, thr=params.thr)

        # meriod dt/2 step 2
        merid(theta, nflux, dt/2, params.merid)

        # diffr dt/4 endpoint
        differential(phi, theta, flux, nflux, dt/4, params.diffr,
                     mode=params.mode_d, thr=params.thr)

        # active region inflow towards regions of strong flux density

        # test for source collisions

        # track unsigned flux history

        # fragment concentrations

        # assimilate magnetogram data

        # add sources
        add_sources()
        
        # forecasting

        # save timestep

        # print stuff

    # finish
        

if __name__ == "__main__":
    print("coming soon")
    theta, phi, flux, rng, params = init_sim()
    loop(theta, phi, flux, rng, params)
