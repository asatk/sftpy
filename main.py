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
from sftpy.emerge.regions import BipoleRegion, MagneticRegion
from sftpy.emerge import PlageNests
from sftpy.fragment import Fragment
from sftpy.initialize import InitTwo
from sftpy.mflow import MF2
from sftpy.misc.carrington import CarringtonRotation
from sftpy.rwalk import RW0, RW2

from sftpy.misc import WrapPhi, WrapTheta
from sftpy.util import Logger, Timestep, MapSaver
from sftpy.util.logger import TimedLogger
from sftpy.viz import plot_aflux, anim_map_with_flux



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

    # bipole orientation
    joy = rc["schrijver.joy"]
    joy_width = rc["schrijver.joy_width"]
    joy_fold = rc["schrijver.joy_fold"]
    sjzero = rc["schrijver.sjzero"]
    max_lat = rc["schrijver.max_lat"]
    lat_width = rc["schrijver.lat_width"]
    lat_fold = rc["schrijver.lat_fold"]
    turbulent = rc["schrijver.turbulent"]
    psource = rc["schrijver.psource"]
    avefluxd = rc["schrijver.avefluxd"]
    miniflux = rc["schrijver.miniflux"]
    maxflux = rc["schrijver.maxflux"]

    thr = rc["rwalk.thr"]

    rad = rc["physics.rad"]

    thetabins = rc["synoptic.thetabins"]
    phibins = rc["synoptic.phibins"]

    nest_lat_lim = 50.0

    binflux = rc["physics.binflux"]


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
    polarconv = ConvergePolarCaps(t_cycle, time, max_cycles=1)
    cycle = CYC1(time, mult=cycle_mult)
    rwalk_frag = RW0(diffusion=fragdist**2/4/dt)
    ini = InitTwo(nfluxmax)

    bipole = BipoleRegion(
        p=psource,
        minflux=miniflux,
        maxflux=maxflux,
        avefluxd=avefluxd,
        dt=dt,
        turbulent=turbulent,
        lat_width=lat_width,
        lat_fold=lat_fold,
        joy=joy,
        joy_width=joy_width,
        joy_fold=joy_fold,
        sjzero=sjzero,
        rad=rad,
        binflux=binflux,
        mode_ar=True,
        mode_eph=True,
        loglvl=0
    )

    plagenests = PlageNests(
        phibins=phibins,
        thetabins=thetabins,
        binflux=binflux,
        avefluxd=avefluxd,
        thr=thr,
        nest_lat_lim=nest_lat_lim,
        loglvl=0,
    )

    decay = Decay()
    rwalk = RW2(dt)
    mflow = MF2(dt/2)
    dflow1 = DF2(dt/4)
    dflow2 = DF2(dt/2)
    collide = COL2(loglvl=0)
    fragment = Fragment(rwalk_frag, loglvl=0)
    bmr = BMRSchrijver(cycle=cycle, region=bipole, nest=plagenests, dt=dt,
                       nfluxmax=nfluxmax, loglvl=0)

    # initialize simulation
    phi, theta, flux, nflux = ini.init()
    saver.checkpoint(phi, theta, flux, nflux)

    timed_logger.clock_start("sim", "Simulation begins:")
    for i in range(1, nstep + 1):

        elapsed = time.getdays() + dt / 86400

        timed_logger.log(loglvl, f"[{i-1}] t = {elapsed/365.25:.03g} yr ({elapsed} d)")
        timed_logger.clock_start("iter", f"[{i-1}] START")

        if ((nstep - (i - 1)) < nstepsfullres) and bipole.mode_ar:
            correction = correction / ff
            dt = dt / ff
            bipole.mode_eph = True

        # polar converge -- remove half of all concentrations after half cycle
        nflux = polarconv.converge(phi, theta, flux, nflux)

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

        # timed_logger.clock_start("col")
        nflux = collide.collide(phi, theta, flux, nflux)
        # timed_logger.clock_stop("col", "collision: ")

        nflux = fragment.fragment(phi, theta, flux, nflux)

        # TODO make it possible to simulate N cycles by having a list of cycle
        # TODO objects rather than specific cycles behave specific ways
        # TODO this way `emerge` can work multiple cycles at once and can
        # TODO simulate other longer-term cycles
        phi, theta, flux, nflux = bmr.emerge(phi, theta, flux, nflux)

        # timed_logger.log(loglvl, "-" * 85)
        timed_logger.log(loglvl,
                   f"\ttotal nflux: {nflux:6d}\t\t" + \
                   f"total flux: {np.sum(np.abs(flux[:nflux]))*1e18:.03g} Mx\t\t" + \
                   f"net flux: {np.sum(flux[:nflux])} Mx")

        timed_logger.clock_stop("iter", f"[{i-1}] END")
        timed_logger.clock_check("sim", "Simulation elapsed time: ")

        time.step()
        saver.checkpoint(phi, theta, flux, nflux)

    '''
        # arinflow
        # assimilate magnetogram data
        # add sources
        # forecasting
    '''

    # finish
    timed_logger.clock_stop("sim", "Simulation completed in ")
    timed_logger.clock_start("sim", "Simulation finished: ")

    # tdelta_col = timed_logger.clock_delta("col", "Time spent in `col`:")
    # tdelta_iter = timed_logger.clock_delta("iter", "Time spent in `iter`:")

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

    outfile_maps = outpath + "/maps"
    outfile_flux = outpath + "/flux"

    saver = loop()
    saver.save(outfile_maps)
    maps = saver.maps

    plot_aflux(
        maps,
        fname=outfile_flux,
        show=True)

    anim_map_with_flux(
        maps,
        flux_thresh=100,
        ms=250,
        fpath=outpath,
        format="gif",
        show=True)
