"""
The `collide` module defines the behavior of colliding and coalescing flux
concentrations. The behavior of the original Schrijver+ model is found in
`collide.py` as the Components `COL1` and COL2`, corresponding to each
of the 'collide modes'.
"""

from .collide import Collide
from .collide import COL1
from .collide import COL2

__all__ = [
        "Collide",
        "COL1",
        "COL2",
]
