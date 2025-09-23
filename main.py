"""
Main script that runs the Solar Flux Transport model for Python.


"""

import numpy as np

from sftpy import simrc as rc

from sftpy.collide import COL1
from sftpy.cycle import CYC1
from sftpy.decay import Decay
from sftpy.dflow import DF2
from sftpy.emerge import BMRSchrijver
from sftpy.fragment import Fragment
from sftpy.initialize import InitTwo
from sftpy.mflow import MF2
from sftpy.rwalk import RW0, RW2

from sftpy.misc import WrapPhi, WrapTheta
from sftpy.util import Logger, synoptic_map, Timestep
from sftpy.viz import plot_syn, anim_syn, plot_aflux


def loop():

    loglvl = rc["general.loglvl"]
    dt = rc["general.dt"]
    savestep = rc["general.savestep"]
    nstep = rc["general.nstep"]
    fragdist = rc["fragment.fragdist"]

    # this is not const in the sim -- make local once loaded into mem
    inv_pol = rc["cycle.inv_pol"]

    nfluxmax = rc["general.nfluxmax"]
    # TODO confirm that this is same
    source = rc["cycle.mult"]

    # logger
    logger = Logger(loglvl, "[loop]")

    # define computation components
    time = Timestep()
    pwrap = WrapPhi()
    twrap = WrapTheta()
    cycle = CYC1(time)
    rwalk_frag = RW0(diffusion=fragdist**2/4/dt)
    ini = InitSimple(nfluxmax)

    decay = Decay()
    rwalk = RW2(dt)
    mflow = MF2(dt/2)
    dflow1 = DF2(dt/4)
    dflow2 = DF2(dt/2)
    collide = COL1(loglvl=2)
    fragment = Fragment(rwalk_frag)
    bmr = BMRSchrijver(nfluxmax, loglvl=0)

    # save synoptic maps at regular intervals
    synoptic_all = np.empty(((nstep - 1) // savestep + 1, 360, 180), dtype=np.int64)

    # initialize simulation
    phi, theta, flux, nflux = ini.init()

    # save initial step
    synoptic_save = synoptic_map(phi, theta, flux, nflux)
    synoptic_all[0] = synoptic_save

    logger.clockstart("sim", "Simulation begins:")
    for i in range(1, nstep):

        time.step()
        logger.log(1, f"[{i}] t = {time/86400/365:.02g} yr")
        logger.clockstart("iter")

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
        logger.clockstart("col")
        nflux = collide.collide(phi, theta, flux, nflux)
        logger.clockstop("col", "collide: ")
        nflux = fragment.fragment(phi, theta, flux, nflux)

        source_str, latsource = cycle.cycle()
        source_str *= inv_pol * 2 - 1

        logger.clockstart("emerge")
        phi, theta, flux, nflux = bmr.emerge(phi, theta, flux, nflux, source_str, latsource, synoptic)
        logger.clockstop("emerge", "emerge: ")

        if i % savestep == 0:
            synoptic_save = synoptic_map(phi, theta, flux, nflux)
            synoptic_all[i//savestep] = synoptic_save


        logger.log(1, f"nflux {nflux}")

        # logger.plot(1, "imshow", synoptic.T)
        if loglvl >= 2:
            plot_syn(phi, theta, flux, nflux, name=f"Step {i}" +\
                     f"({time/86400:.01f} d)", show=True)

        logger.clockstop("iter", f"[{i}]")


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
        # difftest -- thr syn map at 3G
        # TODO this should be another mode and not source-dependent
        # from movecharges:
        # if source , 0 (testruns for diffusion models) then difftest reutnrs
        # the flux in the synmap above a thr of 3G.
        if source < 0:
            # isn't np.fabs redundant?
            synoptic_abs = np.fabs(synoptic)
            temp = 10 / (binflux / 1.4752)
            # where is difftest used
            difftest = np.sum(synoptic_abs[synoptic_abs > temp]) * self._dt
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
    logger.clockstop("sim", "Simulation completed in ")
    logger.clockstart("sim", "Simulation finished: ")

    plot_syn(phi, theta, flux, nflux, name="Final Stellar Surface", show=True)

    return synoptic_all

        

if __name__ == "__main__":
    outfile = rc["general.outfile"]

    synoptic_all = loop()
    np.save(outfile, synoptic_all)
    plot_aflux(synoptic_all, show=True)
    anim_syn(synoptic_all, flux_thresh=100, ms=100, show=True)
