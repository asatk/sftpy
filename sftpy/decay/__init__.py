"""
Decay components reduce the strength of magnetic flux on the stellar surface.
"""

from . import constants
from ._decay import SchrijverDecay

__all__ = [
    "constants",
    "SchrijverDecay"
]
