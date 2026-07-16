"""
Emerge components describe the creation of spots in correspondence with
stellar activity levels.
"""

from ._bmr import BipoleRegion
from ._nesting import PlageNests

from .bmr import BMREmerge
from .bmr import BMRSchrijver


__all__ = [
    "BipoleRegion",
    "PlageNests",
    "BMREmerge",
    "BMRSchrijver"
]
