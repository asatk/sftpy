"""
The `interact` module defines the behavior of colliding and coalescing flux
concentrations. The behavior of the original Schrijver+ model is found in
`interact.py` as the Components `COL1` and COL2`, corresponding to each
of the 'interact modes'.
"""

from . import constants
from ._collide import SchrijverCollide

__all__ = [
        "constants",
        "SchrijverCollide",
]
