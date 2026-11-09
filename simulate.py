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
from sftpy.sim import Data, RecurringEvent, Simulation, Timeline, Logger
from sftpy.sim.constants import BINFLUX, T_START
from sftpy.util import ConvergePolarCaps, SaveMap, WrapPhi, WrapTheta

from sftpy.sim import Actor, Component
from sftpy.util.constants import SAVE_FREQ

dt: float = 1 * 86400.

# from sftpy.util import constants as uconsts
# uconsts.SAVE_FREQ = dt
# TODO setting constants in this way doesn't work
# NEED TO HAVE SEPARATE SIMULATION CONFIGURATION FILE
# important for saving and analysis of sim data


# TODO have a useful logger event/actor class for user
class LogStepEvent(RecurringEvent):

    def __init__(self,
                 time: Timeline,
                 data: Data):
        super().__init__(time, data, t_first=T_START, t_delta=SAVE_FREQ)
        logger = Logger(0, "event")
        self._logger = logger
        logger.clock_start("total")
        logger.clock_start("elapsed")
        self._last_step = -1


    def simulate(self):
        step = self._time.step
        time_y = self._time.time("y")

        total_m = self._logger.clock_stop("total").total_seconds() / 60
        elapsed_s = self._logger.clock_stop("elapsed").total_seconds()

        elapsed_s_per = np.float64(elapsed_s) / (step - self._last_step)

        # format flux string nicely
        flux = self._data.flux
        nflux = self._data.nflux
        flux_total = float(np.sum(np.abs(flux[:nflux])) * BINFLUX)
        if flux_total == 0.0:
            flux_exp = 0
        else:
            flux_exp = int(np.log10(flux_total))
        flux_dec = flux_total / 10**flux_exp

        print(f"[{step:>6d}] --- TOTAL: {total_m:.2f}m --- ELAPSED: {elapsed_s_per:.2f}s / step --- TIME: {time_y:>.2f} yr --- FLUX: {flux_dec:.2f} 10^{flux_exp + 18} Mx --- NFLUX: {nflux:>6d}")

        self._logger.clock_start("elapsed")
        self._last_step = step



def setup_sim() -> Simulation:
    # TODO this setup will eventually have a graphical counterpart
    nstep: int = 40001

    output_dir = "./runs/nov07fivecycles/"
    # uconsts.SAVE_FREQ = 1 * dt

    # Primary simulation timeline
    time = Timeline([(nstep, dt)])

    # Secondary timeline for fractional timesteps with differential rotation
    time_dr = Timeline([(1, dt / 4), (1, dt / 2), (1, dt / 4)] * nstep)

    # Secondary timeline for fractional timesteps with meridional flow
    time_mf = Timeline([(nstep * 2, dt / 2)])

    # Simulation data shared amongst all components
    data = Data()

    # Image processing for facilitating large-scale observational effects
    impro = ImageProcessor(time, data)

    # Solar activity half-cycle
    cycle1 = SchrijverCycle(time, data)

    # opposite-polarity half-cycle
    cycle2 = SchrijverCycle(time, data, phase_const=0.5, multiplier=-1.0)

    # Plage Nesting in Schrijver-style spot emergence
    nest = PlageNests(time, data, impro)

    # Differential Rotation
    rot = SinPolyDR(time_dr, data)

    # Meridional Flow profile
    flow = SinTaperMF(time_mf, data)

    # Regions emerged by simulation
    regions = [
        SchrijverBipole(time, data, cycle1),
        SchrijverEphemeral(time, data, cycle1),
        SchrijverBipole(time, data, cycle2),
        SchrijverEphemeral(time, data, cycle2),
    ]

    # log steps
    log_step_evt = LogStepEvent(time, data)

    # initialization of simulation
    init_evt = InitTwo(time, data)

    # converge polar caps after first half-cycle completes
    converge_caps_evt = ConvergePolarCaps(time, data)

    # save simulation results
    save_evt = SaveMap(time, data, output_dir)

    sequence = [
        log_step_evt,
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
        SchrijverCollide(time, data),
        Fragment(time, data),
        SchrijverEmerge(time, data, regions, nest),
        WrapPhi(time, data),
        WrapTheta(time, data),
        save_evt,
        time
    ]

    s = Simulation(time, data, sequence)

    s.register_event(log_step_evt)
    s.register_event(init_evt)
    s.register_event(converge_caps_evt)
    s.register_event(save_evt)

    return s



if __name__ == "__main__":
    sim = setup_sim()
    sim.run()