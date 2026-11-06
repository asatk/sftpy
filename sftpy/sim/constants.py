# time at which the simulation begins (yr)
T_START: float = 0.0

# random seed
SEED = 0x2026

# amount of flux counted in each discrete bin (10^18 Mx)
BINFLUX: float = 1.0

# stellar radius (km)
STAR_RADIUS: float = 695800.0

# initial maximum capacity for data arrays
NFLUXMAX: int = 250_000

# default logging level of a new instance of a Component
LOGGING_LEVEL: int = 0

# TODO multithreading not yet implemented for components
# number of threads for spawned for sub-processes
NUM_THREADS: int = 4