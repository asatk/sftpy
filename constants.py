"""
This module contains all general physical and other constants
"""

###      ###
# Activity #
###      ###

from activity import ActivityConstant
ActivityConstant.multiplier = 1.0
ActivityConstant.latsource = 25.0
ActivityConstant.loglvl = 0

from activity import SchrijverCycle
SchrijverCycle.latlo = 0.0
SchrijverCycle.lathi = 25.0
SchrijverCycle.period = 21.9
SchrijverCycle.ovr = 3.0
SchrijverCycle.peak = 4.0
SchrijverCycle.phase = 0.0
SchrijverCycle.multiplier = 1.0
SchrijverCycle.loglvl = 0

from activity import ModulatedSchrijverCycle
ModulatedSchrijverCycle.boxcar_width = 10
ModulatedSchrijverCycle.intep_order = 3
ModulatedSchrijverCycle.latlo = 0.0
ModulatedSchrijverCycle.lathi = 25.0
ModulatedSchrijverCycle.period = 21.9
ModulatedSchrijverCycle.ovr = 3.0
ModulatedSchrijverCycle.peak = 4.0
ModulatedSchrijverCycle.phase = 0.0
ModulatedSchrijverCycle.multiplier = 1.0
ModulatedSchrijverCycle.loglvl = 0

from activity import SolarRecordsCycle
SolarRecordsCycle.boxcar_width = 10
SolarRecordsCycle.intep_order = 3
SolarRecordsCycle.latlo = 0.0
SolarRecordsCycle.lathi = 25.0
SolarRecordsCycle.period = 21.9
SolarRecordsCycle.ovr = 3.0
SolarRecordsCycle.peak = 4.0
SolarRecordsCycle.multiplier = 1.0
SolarRecordsCycle.loglvl = 0



###       ###
# Component #
###       ###
from sim._abcs import Component
Component.logging = 0



###   ###
# Decay #
###   ###

from decay import SchrijverDecay
# half life of flux concentrations (>1000 years yields no decay)
SchrijverDecay.t_decay = 1000.0
SchrijverDecay.loglvl = 0



###    ###
# Emerge #
###    ###
from emerge import constants as emerge_constants
emerge_constants.AR_BFIELD = 180.0

from emerge import SchrijverEmerge
SchrijverEmerge.logging = 0

from emerge import SchrijverAssimilate
SchrijverAssimilate.logging = 0

from emerge import SchrijverGradual
SchrijverGradual.logging = 0

###             ###
# Emerge: Regions #
###             ###

from regions import SchrijverBipole
Sc
SchrijverBipole.logging = 0



###      ###
# Fragment #
###      ###

from fragment import Fragment
# random walk distance for fragmented concentrations
Fragment.fragdist = 5000.0
Fragment.multiplier = 1.0
Fragment.loglvl = 0



###      ###
# Interact #
###      ###

from interact import SchrijverCollide
SchrijverCollide.collide_radius = 4200.
SchrijverCollide.nskips = 25
SchrijverCollide.loglvl = 0



###         ###
# Random Walk #
###         ###

from diffusion import ConstantRW
# diffusion coefficient for flux dispersal (km^2/s)
ConstantRW.diffusion = 300.0
ConstantRW.loglvl = 0

from diffusion import ConstantRW
ConstantRW.diffusion = ConstantRW.diffusion
ConstantRW.loglvl = ConstantRW.loglvl

from diffusion import LocalFluxDensityRW
LocalFluxDensityRW.rwalk1_flux_threshold = 40.0
LocalFluxDensityRW.diffusion = ConstantRW.diffusion
LocalFluxDensityRW.loglvl = ConstantRW.loglvl

from diffusion import FluxRW
FluxRW.diffusion = ConstantRW.diffusion
FluxRW.loglvl = ConstantRW.loglvl
