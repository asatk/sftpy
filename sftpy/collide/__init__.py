"""
The `collide` module defines the behavior of colliding and coalescing flux
concentrations. The behavior of the original Schrijver+ model is found in
`collide.py` as the Components `COL1`, COL2`, and `COL3`, corresponding to each
of the 'collide modes'.
"""

from .collide import Collide, COLNone
from .collide import COL1
from .collide import COL2
from .collide import COL3

__all__ = [
        "Collide",
        "COLNone",
        "COL1",
        "COL2",
        "COL3"
]
