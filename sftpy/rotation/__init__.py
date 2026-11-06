"""
Differential rotation components move flux across longitudes (along phi axis)
at different speeds for different co-latitudes (thetas).
"""

from . import constants
from ._carrington import CarringtonRotation
from ._differential import SinPolyDR
from ._differential import SinPolyTaperDR
from ._differential import SinPolyExpandingFluxDR

__all__ = [
        "constants",
        "CarringtonRotation",
    "SinPolyDR",
    "SinPolyTaperDR",
    "SinPolyExpandingFluxDR",
]
