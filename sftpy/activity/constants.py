# strength of activity relative to the average Sun (relative)
MULTIPLIER: float = 1.0

###     ###
# SECULAR #
###     ###

# average emergence latitude (deg)
SEC_LAT: float = 25.0

###    ###
# CYCLIC #
###    ###

# average emergence latitude at cycle start (deg)
CYC_LAT_START: float = 25.0
# average emergence latitude at cycle end (deg)
CYC_LAT_END: float = 0.0
# activity cycle duration (yr)
CYC_PERIOD: float = 21.9
# overlap between consecutive (half-) cycles (yr) -- no more than half activity period
CYC_OVERLAP: float = 3.0
# time at which sunspot number and total flux peaks (yr)
CYC_PEAK: float = 4.0
# time offset as a fraction of the cycle period
CYC_PHASE_CONST: float = 0.0
# width of smoothing window (number of bins)
SMOOTH_WIDTH: int = 10
# polynomial order of interpolation
INTERP_ORDER: int = 3