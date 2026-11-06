import abc
import numpy as np

from sftpy.activity import SchrijverCycle
from sftpy.diffusion import FluxRW
from sftpy.emerge import SchrijverEmerge, PlageNests
from sftpy.regions import SchrijverBipole, SchrijverEphemeral
from sftpy.flow import SinTaperMF
from sftpy.fragment import Fragment
from sftpy.imgproc import ImageProcessor
from sftpy.initialize import InitTwo
from sftpy.interact import SchrijverCollide
from sftpy.rotation import CarringtonRotation, SinPolyDR
from sftpy.sim import Data, RecurringEvent, Simulation, Timeline
from sftpy.sim.constants import BINFLUX, T_START
from sftpy.util import ConvergePolarCaps, SaveMap, WrapPhi, WrapTheta

from sftpy.sim import Actor, Component




dt: float = 86400.

from sftpy.util import constants as uconsts
uconsts.SAVE_FREQ = dt

class Logger(Actor, Component, metaclass=abc.ABCMeta):
    def __init__(self,
                 time: Timeline,
                 data: Data):
        super().__init__(time, data)

class LogStep(Logger):

    def simulate(self):
        step = self._time.step
        flux = self._data.flux
        nflux = self._data.nflux
        flux_total = float(np.sum(np.abs(flux[:nflux])) * BINFLUX)
        if flux_total == 0.0:
            flux_exp = 0
        else:
            flux_exp = int(np.log10(flux_total))
        flux_dec = flux_total / 10**flux_exp
        print(f"[{step:>5d}] --- FLUX: {flux_dec:.2f} 10^{flux_exp + 18} Mx --- NFLUX: {nflux:>6d}")


class CheckNFLUX(Logger):
    def __init__(self,
                 time: Timeline,
                 data: Data):
        super().__init__(time, data)
        self._nflux = data.nflux

    @property
    def nflux(self):
        return self._nflux

    def simulate(self) -> None:
        self._nflux = self._data.nflux

class LogDiffNFLUX(Logger):
    def __init__(self,
                 time: Timeline,
                 data: Data,
                 checker: CheckNFLUX):
        super().__init__(time, data)
        self._checker = checker

    def simulate(self) -> None:
        nflux_old = self._checker.nflux
        nflux = self._data.nflux
        diff = nflux - nflux_old
        print(f"NFLUX --- {diff:>+6d}")



def setup_sim() -> Simulation:
    # TODO this setup will eventually have a graphical counterpart
    nstep: int = 1000

    output_dir = "./runs/nov05run01/"
    uconsts.SAVE_FREQ = 1 * dt

    # Primary simulation timeline
    time = Timeline([(nstep, dt)])

    # Secondary timeline for fractional timesteps with differential rotation
    time_dr = Timeline([(1, dt / 4), (1, dt / 2), (1, dt / 4)] * nstep)

    # Secondary timeline for fractional timesteps with meridional flow
    time_mf = Timeline([(2, dt / 2)] * nstep)

    # Simulation data shared amongst all components
    data = Data()

    # Image processing for facilitating large-scale observational effects
    impro = ImageProcessor(time, data)

    # Solar activity cycle
    activity = SchrijverCycle(time, data)

    # Plage Nesting in Schrijver-style spot emergence
    nest = PlageNests(time, data, impro)

    # Differential Rotation
    rot = SinPolyDR(time_dr, data)

    # Meridional Flow profile
    flow = SinTaperMF(time_mf, data)

    # Regions emerged by simulation
    regions = [
        SchrijverBipole(time, data, activity),
        SchrijverEphemeral(time, data, activity),
    ]

    # initialization of simulation
    init_evt = InitTwo(time, data)

    # converge polar caps after first half-cycle completes
    converge_caps_evt = ConvergePolarCaps(time, data)

    # save simulation results
    save_evt = SaveMap(time, data, output_dir)



    check_nflux = CheckNFLUX(time, data)
    log_dnflux = LogDiffNFLUX(time, data, check_nflux)



    sequence = [
        LogStep(time, data),
        init_evt,
        converge_caps_evt,
        FluxRW(time, data),
        rot,
        time_dr,
        flow,
        time_mf,
        rot,
        time_dr,
        flow,
        time_mf,
        rot,
        time_dr,
        CarringtonRotation(time, data),
        # check_nflux,
        SchrijverCollide(time, data),
        # log_dnflux,
        # check_nflux,
        Fragment(time, data),
        # log_dnflux,
        # check_nflux,
        SchrijverEmerge(time, data, regions, nest),
        # log_dnflux,
        WrapPhi(time, data),
        WrapTheta(time, data),
        save_evt,
        time,
    ]

    s = Simulation(time, data, sequence)

    s.register_event(init_evt)
    s.register_event(converge_caps_evt)
    s.register_event(save_evt)


    return s



if __name__ == "__main__":
    sim = setup_sim()
    sim.run()