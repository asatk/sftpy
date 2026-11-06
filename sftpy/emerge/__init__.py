"""
Emerge components describe the creation of spots in correspondence with
stellar activity levels.
"""

from . import constants

from ._nesting import PlageNests

from ._emerge import SchrijverAssimilate
from ._emerge import SchrijverEmerge

from ._gradual import EmergeGradual

__all__ = [
    "constants",
    "PlageNests",
    "SchrijverAssimilate",
    "SchrijverEmerge",
    "EmergeGradual"
]
