"""
Main script that runs the Solar Flux Transport model for Python.


"""

import numpy as np

from sftpy import simrc as rc

from sftpy.collide import COL2
from sftpy.cycle import CYC1, ConvergePolarCaps
from sftpy.decay import Decay
from sftpy.dflow import DF2
from sftpy.emerge import BMRSchrijver
from sftpy.fragment import Fragment
from sftpy.initialize import InitTwo
from sftpy.mflow import MF2, MF4
from sftpy.misc.carrington import CarringtonRotation
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

    t_cycle = rc["cycle.period"]


    # logger
    logger = Logger(loglvl, "[loop]")

    # define computation components
    time = Timestep()
    pwrap = WrapPhi()
    twrap = WrapTheta()
    crot = CarringtonRotation(dt)
    polarconv = ConvergePolarCaps(t_cycle, time)
    cycle = CYC1(time)
    rwalk_frag = RW0(diffusion=fragdist**2/4/dt)
    ini = InitTwo(nfluxmax)

    decay = Decay()
    rwalk = RW2(dt)
    # mflow = MF4(cycle, dt/2)
    mflow = MF2(dt/2)
    dflow1 = DF2(dt/4)
    dflow2 = DF2(dt/2)
    collide = COL2(loglvl=0)
    fragment = Fragment(rwalk_frag)
    bmr = BMRSchrijver(nfluxmax, loglvl=0)

    # save synoptic maps at regular intervals
    synoptic_all = np.empty(((nstep - 1) // savestep + 1, 360, 180), dtype=np.int64)

    # initialize simulation
    phi, theta, flux, nflux = ini.init()

    # save initial step
    synoptic_save = synoptic_map(phi, theta, flux, nflux)
    synoptic_all[0] = synoptic_save

    logger.clock_start("sim", "Simulation begins:")
    for i in range(1, nstep):

        time.step()
        if i % savestep == 0:
            lognum = loglvl
        else:
            lognum = loglvl + 1

        logger.log(lognum, f"[{i}] t = {time/86400/365:.02g} yr")
        logger.clock_start("iter")

        # polar converge -- remove half of all concentrations after half cycle
        # nflux = polarconv.converge(phi, theta, flux, nflux)

        nflux = decay.decay(phi, theta, flux, nflux)
        synoptic = synoptic_map(phi, theta, np.fabs(flux), nflux)
        synoptic = rwalk.move(phi, theta, flux, nflux, synoptic)
        dflow1.move(phi, theta, flux, nflux)
        mflow.move(theta, nflux)
        dflow2.move(phi, theta, flux, nflux)
        mflow.move(theta, nflux)
        dflow1.move(phi, theta, flux, nflux)
        crot.move(phi, nflux)
        pwrap(phi, nflux)
        twrap(phi, theta, nflux)
        nflux = collide.collide(phi, theta, flux, nflux)
        nflux = fragment.fragment(phi, theta, flux, nflux)

        source_str, latsource = cycle.cycle()
        source_str *= inv_pol * 2 - 1

        phi, theta, flux, nflux = bmr.emerge(phi, theta, flux, nflux, source_str, latsource, synoptic)

        # TODO introduce checkpointer
        if i % savestep == 0:
            synoptic_save = synoptic_map(phi, theta, flux, nflux)
            synoptic_all[i//savestep] = synoptic_save
            logger.clock_check("iter", f"[{i}]")
            logger.clock_check("sim", "Simulation elapsed time: ")

        logger.log(lognum, f"nflux {nflux}")

    '''

        """
        # scale control params back
        if params.nstep - i < params.nstepfullres and as_specified == 0:
            correction /= params.ff
            dt /= params.ff
            as_specified = True
        """
        
        # calculate synoptic map
        synoptic = synoptic_map(phi, theta, np.fabs(flux), nflux)
        # arinflow
        # assimilate magnetogram data
        # add sources
        # forecasting
    '''

    # finish
    logger.clock_stop("sim", "Simulation completed in ")
    logger.clock_start("sim", "Simulation finished: ")

    plot_syn(phi, theta, flux, nflux, name="Final Stellar Surface", show=True)

    return synoptic_all

        

if __name__ == "__main__":
    outfile = rc["general.outfile"]

    synoptic_all = loop()
    np.save(outfile, synoptic_all)
    plot_aflux(synoptic_all, show=True)
    anim_syn(synoptic_all, flux_thresh=100, ms=100, show=True)
