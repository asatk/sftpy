from .cycle import cyl_t
from .cycle import cyl_mult

from .cycle import cycle_0
from .cycle import cycle_1
from .cycle import cycle_2
from .cycle import cycle_3
from .cycle import cycle_4

cycle_modes = [
    cycle_0,
    cycle_1,
    cycle_2,
    cycle_3,
    cycle_4,
]

__all__ = [
    "cycle_modes",
    "cyl_t",
    "cyl_mult"
]
