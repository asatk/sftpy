"""
Fragment components reduce the flux of spots by breaking them up into
multiple spots with less flux.
"""

from . import constants

from ._fragment import Fragment

__all__ = [
    "constants",
    "Fragment"
]
