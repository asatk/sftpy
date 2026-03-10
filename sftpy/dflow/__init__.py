"""
Differential rotation components move flux across longitudes (along phi axis)
at different speeds for different co-latitudes (thetas).
"""

from .dflow import DifferentialFlow
from .dflow import DF1
from .dflow import DF2
from .dflow import DF3
from .dflow import DF4

__all__ = [
        "DifferentialFlow",
        "DF1",
        "DF2",
        "DF3",
        "DF4",
]
