import numpy.random as npr

from .constants import SEED

# set global generator's seed
rng = npr.default_rng(seed=SEED)
npr.seed(SEED)