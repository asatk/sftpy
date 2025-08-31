"""
Data visualization module
"""

from .maps import plot_syn, anim_syn
from .hists import plot_aflux, plot_hist, plot_lat

__all__ = [
    "plot_syn",
    "anim_syn",
    "plot_aflux",
    "plot_hist",
    "plot_lat"
]
