"""
Initialize components start the SFT simulation with some pattern of spots on
the stellar surface.
"""

from . import constants
from ._initialize import Initialize
from ._simple import InitTwo

__all__ = [
    "constants",
    "Initialize",
    "InitTwo",
]
