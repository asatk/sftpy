"""
RandomWalk components move sources randomly by using a model of diffusion on
the stellar surface.
"""

from .rwalk import RandomWalk
from .rwalk import RW0
from .rwalk import RW1
from .rwalk import RW2

__all__ = [
    "RandomWalk",
    "RW0",
    "RW1",
    "RW2"
]
