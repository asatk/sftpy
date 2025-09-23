"""
Initialize components start the SFT simulation with some pattern of spots on
the stellar surface.
"""

from .initialize import Initialize
from .simple import InitOne, InitTwo

__all__ = [
    "Initialize",
    "InitOne",
    "InitTwo",
]
