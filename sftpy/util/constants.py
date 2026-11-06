from ..activity.constants import CYC_PERIOD

# number of seconds between saves
SAVE_FREQ: int = 10 * 86400

# time at which the polar caps are scheduled to converge (s)
T_CONVERGE = CYC_PERIOD / 2 / 86400 / 365.25