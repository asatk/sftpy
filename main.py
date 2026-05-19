"""
Main script that runs the Solar Flux Transport model for Python.


"""

import numpy as np
import os

from sftpy import simrc as rc

from sftpy.collide import COL2
from sftpy.cycle import CYC1, ConvergePolarCaps
from sftpy.decay import Decay
from sftpy.dflow import DF2
from sftpy.emerge import BMRSchrijver
from sftpy.fragment import Fragment
from sftpy.initialize import InitTwo
from sftpy.mflow import MF2
from sftpy.misc.carrington import CarringtonRotation
from sftpy.rwalk import RW0, RW2

from sftpy.misc import WrapPhi, WrapTheta
from sftpy.util import Logger, Timestep, MapSaver
from sftpy.util.logger import TimedLogger
from sftpy.viz import plot_syn, plot_aflux, anim_map_with_flux



def loop():

    loglvl = rc["general.loglvl"]
    dt = rc["general.dt"]
    savestep = rc["general.savestep"]
    nstep = rc["general.nstep"]
    nfluxmax = rc["general.nfluxmax"]
    ff = rc["general.ff"]
    correction = rc["general.correction"]

    fragdist = rc["fragment.fragdist"]

    cycle_mult = rc["cycle.mult"]
    t_cycle = rc["cycle.period"]

    phibins = rc["synoptic.phibins"]
    thetabins = rc["synoptic.thetabins"]

    as_specified = rc["schrijver.as_specified"]


    nstepsfullres = nstep - 1
    nstepslowres = int((nstep - nstepsfullres) / ff)
    nstep = nstepslowres + nstepsfullres

    # timestep
    time = Timestep(dt=dt)

    # logger
    timed_logger = TimedLogger(time, savestep, loglvl, "[loop]")

    # saver
    saver = MapSaver(
        frequency=savestep,
        timestep=time,
        phibins=phibins,
        thetabins=thetabins,
        nsteps=nstep)

    # define computation components
    pwrap = WrapPhi()
    twrap = WrapTheta()
    crot = CarringtonRotation(dt)
    polarconv = ConvergePolarCaps(t_cycle, time)
    cycle = CYC1(time, mult=cycle_mult)
    rwalk_frag = RW0(diffusion=fragdist**2/4/dt)
    ini = InitTwo(nfluxmax)

    decay = Decay()
    rwalk = RW2(dt)
    mflow = MF2(dt/2)
    dflow1 = DF2(dt/4)
    dflow2 = DF2(dt/2)
    collide = COL2(loglvl=0)
    fragment = Fragment(rwalk_frag)
    bmr = BMRSchrijver(cycle=cycle, dt=dt,
                       nfluxmax=nfluxmax, as_specified=as_specified, loglvl=0)

    # initialize simulation
    phi, theta, flux, nflux = ini.init()
    saver.checkpoint(phi, theta, np.abs(flux), nflux)

    timed_logger.clock_start("sim", "Simulation begins:")
    for i in range(1, nstep + 1):

        timed_logger.log(loglvl, f"[{i-1}] t = {time/86400/365:.03g} yr")
        timed_logger.clock_start("iter")

        if ((nstep - (i - 1)) < nstepsfullres) and not bmr.as_specified:
            correction = correction / ff
            dt = dt / ff
            bmr.as_specified = True

        # polar converge -- remove half of all concentrations after half cycle
        # nflux = polarconv.converge(phi, theta, flux, nflux)

        nflux = decay.decay(phi, theta, flux, nflux)
        rwalk.move(phi, theta, flux, nflux)
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
        phi, theta, flux, nflux = bmr.emerge(phi, theta, flux, nflux)

        timed_logger.log(loglvl, f"Signed flux: {np.sum(flux[:nflux])}")
        timed_logger.log(loglvl,
                   f"spots: {nflux}\t" + \
                   f"total flux: {np.sum(np.abs(flux[:nflux]))*1e18:.03g} Mx")

        timed_logger.clock_check("iter", f"[{i-1}]")
        timed_logger.clock_check("sim", "Simulation elapsed time: ")

        time.step()
        saver.checkpoint(phi, theta, np.abs(flux), nflux)

    '''
        # arinflow
        # assimilate magnetogram data
        # add sources
        # forecasting
    '''

    # finish
    timed_logger.clock_stop("sim", "Simulation completed in ")
    timed_logger.clock_start("sim", "Simulation finished: ")

    return saver

        

if __name__ == "__main__":

    outpath = rc["general.outpath"]
    parentpath = outpath[:outpath.rfind("/")]

    if not os.path.isdir(outpath) and os.path.isdir(parentpath):
        print(f"INFO: creating run directory: {outpath}")
        os.mkdir(outpath)

    if os.path.isdir(outpath):
        print(f"INFO: run data will be stored at: {outpath}")
    else:
        print(f"FATAL: output directory does not exist: {outpath}")
        exit(1)

    outfile = outpath + "/maps.npy"

    saver = loop()
    saver.save(outfile)

    maps = saver.maps
    plot_aflux(maps, show=True)
    anim_map_with_flux(
        maps,
        flux_thresh=100,
        ms=250,
        fpath=outpath,
        format="gif",
        show=True)
