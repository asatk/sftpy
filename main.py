"""
Main script that runs the Solar Flux Transport model for Python.


"""

from matplotlib import pyplot as plt
import numpy as np

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
from sftpy.util import Logger, MapMaker, Timestep
from sftpy.viz import plot_syn, plot_aflux, anim_map_with_flux


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

    phibins = rc["synoptic.phibins"]
    thetabins = rc["synoptic.thetabins"]


    # logger
    logger = Logger(loglvl, "[loop]")

    # define computation components
    time = Timestep(dt=dt)
    pwrap = WrapPhi()
    twrap = WrapTheta()
    crot = CarringtonRotation(dt)
    map_maker = MapMaker(phibins, thetabins)
    polarconv = ConvergePolarCaps(t_cycle, time)
    cycle = CYC1(time)
    rwalk_frag = RW0(diffusion=fragdist**2/4/dt)
    ini = InitTwo(nfluxmax)

    decay = Decay()
    rwalk = RW2(dt)
    mflow = MF2(dt/2)
    dflow1 = DF2(dt/4)
    dflow2 = DF2(dt/2)
    collide = COL2(loglvl=0)
    fragment = Fragment(rwalk_frag)
    bmr = BMRSchrijver(map_maker=map_maker, dt=dt, nfluxmax=nfluxmax, loglvl=0)

    # save synoptic maps at regular intervals
    map_series = np.empty((nstep // savestep + 1, 360, 180), dtype=np.int64)

    # initialize simulation
    phi, theta, flux, nflux = ini.init()

    # save initial step
    map_curr = map_maker.make_Carrington_map(phi, theta, flux, nflux)
    map_series[0] = map_curr

    logger.clock_start("sim", "Simulation begins:")
    for i in range(1, nstep + 1):

        time.step()
        if i % savestep == 0:
            lognum = loglvl
        else:
            lognum = loglvl + 1

        logger.log(lognum, f"[{i}] t = {time/86400/365:.03g} yr")
        logger.clock_start("iter")

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

        source_str, latsource = cycle.cycle()
        source_str *= inv_pol * 2 - 1

        phi, theta, flux, nflux = bmr.emerge(
            phi, theta, flux, nflux, source_str, latsource)

        # if i % savestep == 0:
        #     plot_syn(phi, theta, flux, nflux)
        #
        #     plt.show()
        #     tempbins = np.arcsin(np.linspace(-1+1e-5, 1-1e-5, 180, endpoint=True)) + np.pi/2
        #     logger.plot(1, "hist", theta[:nflux],
        #                 weights=np.abs(flux[:nflux]), bins=tempbins, range=(0, np.pi),
        #                 histtype="step")
        #     logger.plot(1, "title", "colatitude")
        #     logger.pshow(1)

        # TODO introduce checkpointer
        if i % savestep == 0:
            map_curr = map_maker.make_Carrington_map(phi, theta, flux, nflux)
            map_series[i//savestep] = map_curr
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

    return map_series

        

if __name__ == "__main__":
    outfile = rc["general.outfile"]

    map_series = loop()
    np.save(outfile, map_series)
    plot_aflux(map_series, show=True)
    anim_map_with_flux(map_series, flux_thresh=100, ms=100, format="gif", show=True)
