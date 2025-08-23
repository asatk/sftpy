"""
Model driver
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Callable

from plot import plot_syn, plot_lat, plot_hist, anim_syn

from sftpy.collide import COLNone, COL1, COL2, COL3
from sftpy.cycle import CYCNone, CYC1, CYC2, CYC3, CYC4
from sftpy.decay import Decay
from sftpy.dflow import DFNone, DF1, DF2, DF3, DF4
from sftpy.emerge import BMRSchrijver
from sftpy.mflow import MFNone, MF1, MF2, MF3, MF4
from sftpy.rwalk import RWNone, RW0, RW1, RW2

from sftpy.util import Logger, synoptic_map, Timestep
from sftpy.misc import WrapPhi, WrapTheta

class Params(dict):
    
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

class Driver():

    def __init__(self, comps: list):
        self._comps = comps

    def loop(self):
        for i in range(nstep):
            for comp in self._comps:
                self._comps


def init_sim() -> dict:

    params = Params(
        #dt=86400,   # 24 hrs
        #dt=21600,   # 6 hrs
        dt=3600*3,   # 1 hr
        nstep=30,
        savestep=1,
        seed=0x2025,
        nflux=2,
        nfluxmax=250000,
        t_decay=1,
        inv_pol=1,
        source=1.0,
        phibins=360,
        thetabins=180,
        savelat=False,
        nstepfullres=1000000,
        as_specified=True,
        polarconverge=False,
        remhalf=False,
        correction=1.0,
        ff=1.0,
        outfile="maps.npy",
        loglvl=0)

    rng = np.random.default_rng(seed=params.seed)

    phi = np.zeros(params.nfluxmax, dtype=np.float64)
    phi[0] = np.pi * 20 / 180
    phi[1] = np.pi * 20 / 180
    phi[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

    theta = np.zeros(params.nfluxmax, dtype=np.float64)
    theta[0] = np.pi * 107 / 180
    theta[1] = np.pi *  73 / 180
    theta[:2] += rng.normal(scale=np.pi * 2 / 180, size=2)

    init_flux = 100

    flux = np.zeros(params.nfluxmax, dtype=np.int64)
    flux[0] = init_flux * params.inv_pol
    flux[1] = -init_flux * params.inv_pol

    nflux = 2
    
    params.phi = phi
    params.theta = theta
    params.flux = flux
    params.nflux = nflux
    params.rng = rng
    
    print("initial")
    plot_syn(phi, theta, flux, nflux, name="map-initial.png")

    return params


def loop(params: Params):

    phi = params.phi
    theta = params.theta
    flux = params.flux
    nflux = params.nflux
    rng = params.rng

    nstep = params.nstep
    dt = params.dt
    nfluxmax = params.nfluxmax
    source = params.source
    as_specified = params.as_specified
    correction = params.correction
    savestep = params.savestep
    outfile = params.outfile

    # logger
    logger = Logger(params.loglvl, "[loop]")

    # define computation components
    time = Timestep(dt)
    pwrap = WrapPhi()
    twrap = WrapTheta()
    cycle = CYCNone(time)

    decay = Decay(dt, rng, t_decay=1000)
    rwalk = RWNone(dt, rng)
    mflow = MFNone(dt/2)
    dflow1 = DFNone(dt/4)
    dflow2 = DFNone(dt/2)
    collide = COLNone(dt, rng, correction=correction)
    bmr = BMRSchrijver(dt, rng, nfluxmax)

    # save synoptic maps at regular intervals
    synoptic_all = np.empty(((nstep - 1) // savestep + 1, 360, 180), dtype=np.int64)

    # save initial step
    synoptic_save = synoptic_map(phi, theta, flux, nflux)
    synoptic_all[0] = synoptic_save

    for i in range(1, nstep):

        time.step()
        logger.log(0, f"[{i:d}] time {time:.02e}")

        nflux = decay.decay(phi, theta, flux, nflux)
        synoptic = synoptic_map(phi, theta, np.fabs(flux), nflux)
        synoptic = rwalk.move(phi, theta, flux, nflux, synoptic, source)
        dflow1.move(phi, theta, flux, nflux)
        mflow.move(theta, nflux)
        dflow2.move(phi, theta, flux, nflux)
        mflow.move(theta, nflux)
        dflow1.move(phi, theta, flux, nflux)
        pwrap(phi, nflux)
        twrap(phi, theta, nflux)
        nflux = collide.collide(phi, theta, flux, nflux)

        source_str, latsource = cycle.cycle()
        source_str *= params.inv_pol * 2 - 1

        phi, theta, flux, nflux = bmr.emerge(phi, theta, flux, nflux, source_str, latsource, synoptic)

        if i % savestep == 0:
            synoptic_save = synoptic_map(phi, theta, flux, nflux)
            synoptic_all[i//savestep] = synoptic_save


        logger.log(0, f"nflux {nflux}")
        logger.log(2, f"flux {flux[:nflux]}")
        logger.log(2, f"theta {theta[:nflux]}")
        logger.log(2, f"phi {phi[:nflux]}")


    '''
        """
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
        """

        """
        # scale control params back
        if params.nstep - i < params.nstepfullres and as_specified == 0:
            correction /= params.ff
            dt /= params.ff
            as_specified = True
        """

        """
        # decay field
        # safety catch -- NaN
        """
        
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

        """
        print(f"[{i:8d}|random_walk]")
        synoptic = random_walk(phi, theta, flux, nflux, dt, rng, synoptic,
                               source=source)
        # safety catch -- NaN
        """

        # alternate applying merid and diffr steps
        # active region inflow towards regions of strong flux density
        # test for source collisions

        # track unsigned flux history

        # fragment concentrations

        # assimilate magnetogram data

        # cycle
        source_str, latsource = cycle(time)
        source_str *= params.inv_pol * 2 - 1
        # print


        # add sources

        # forecasting

        # save timestep

        # print stuff
    '''

    # finish

    plot_syn(phi, theta, flux, nflux, name="map-final.png")

    return synoptic_all

        

if __name__ == "__main__":
    outfile = "maps.npy"

    params = init_sim()
    synoptic_all = loop(params)
    np.save(outfile, synoptic_all)
    anim_syn(synoptic_all, params.dt*params.savestep)
    plt.show()
