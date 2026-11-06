"""
Main script that runs the Solar Flux Transport model for Python.


"""

import numpy as np
import os


from sftpy import simrc as rc

from sftpy.interact import SchrijverCollide
from sftpy.activity import CYC1, ConvergePolarCaps
from sftpy.decay import SchrijverDecay
from sftpy.rotation import DF2
from sftpy.emerge import BipoleSchrijver
from regions import SchrijverBipole
from sftpy.emerge import PlageNests
from sftpy.fragment import Fragment
from sftpy.initialize import InitTwo
from sftpy.flow import SinTaperMF
from rotation._carrington import CarringtonRotation
from sftpy.misc import FastForward
from sftpy.diffusion import ConstantRW, FluxRW

from sftpy.misc import WrapPhi, WrapTheta
from sftpy.util import Timeline, MapSaver
from _logger import TimedLogger
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

    cycle_mult = rc["activity.mult"]
    t_cycle = rc["activity.period"]

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

    thr = rc["diffusion.thr"]

    rad = rc["physics.rad"]

    nest_lat_lim = 50.0

    binflux = rc["physics.binflux"]

    # AD Leo
    prot = 161.


    nstepsfullres = nstep - 1
    nstepslowres = int((nstep - nstepsfullres) / ff)
    nstep = nstepslowres + nstepsfullres

    # timestep
    time = Timeline(dt=dt)

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
    crot = CarringtonRotation(dt, crot=prot)
    polarconv = ConvergePolarCaps(t_cycle, time, max_cycles=1)
    cycle = CYC1(time, mult=cycle_mult)
    rwalk_frag = ConstantRW(diffusion=fragdist ** 2 / 4 / dt)
    ini = InitTwo(nfluxmax)

    bipole = SchrijverBipole(
        p=psource,
        flux_min=miniflux,
        flux_max=maxflux,
        ar_bfield=avefluxd,
        dt=dt,
        fraction_trb=turbulent,
        lat_width=lat_width,
        lat_flux_fold=lat_fold,
        joy=joy,
        tilt_width=joy_width,
        tilt_flux_fold=joy_fold,
        sjzero=sjzero,
        rad=rad,
        binflux=binflux,
        flag_act=True,
        flag_eph=True,
        logging=0
    )

    plagenests = PlageNests(
        nphi=phibins,
        ntheta=thetabins,
        binflux=binflux,
        avefluxd=avefluxd,
        thr=thr,
        nest_lat_max=nest_lat_lim,
        loglvl=0,
    )

    decay = SchrijverDecay()
    rwalk = FluxRW(dt)
    mflow = SinTaperMF(dt / 2)
    dflow1 = DF2(dt/4)
    dflow2 = DF2(dt/2)
    collide = SchrijverCollide(logging=0)
    fragment = Fragment(rwalk_frag, logging=0)
    bmr = BipoleSchrijver(cycle=cycle, region=bipole, nest=plagenests, dt=dt,
                          nfluxmax=nfluxmax, loglvl=0)

    # initialize simulation
    phi, theta, flux, nflux = ini.init()
    saver.checkpoint(phi, theta, flux, nflux)

    fastforward = FastForward(
        time,
        bipole,
        ff = 1.0,
        nstepff=nstep - nstepsfullres + 1
    )

    # TODO print all important options and ask user to confirm?

    timed_logger.clock_start("sim", "Simulation begins:")
    for i in range(1, nstep + 1):

        elapsed = time.time_d() + dt / 86400

        timed_logger.log(loglvl, f"[{i-1}] t = {elapsed/365.25:.03g} yr ({elapsed} d)")
        timed_logger.clock_start("iter", f"[{i-1}] START")

        fastforward.slowdown()

        # polar converge -- remove half of all concentrations after half activity
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

        nflux = fragment.simulate(phi, theta, flux, nflux)

        # TODO make it possible to simulate N cycles by having a list of activity
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

        time.advance()
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
