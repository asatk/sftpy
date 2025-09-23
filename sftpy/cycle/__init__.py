"""
Cycle components determine the level of stellar activity or equivalently the
number of star spots to emerge.
"""

from .cycle import Cycle, CYCNone
from .cycle import CYC0
from .cycle import CYC1
from .cycle import CYC2
from .cycle import CYC3
from .cycle import CYC4

__all__ = [
    "Cycle",
    "CYCNone",
    "CYC0",
    "CYC1",
    "CYC2",
    "CYC3",
    "CYC4",
]
