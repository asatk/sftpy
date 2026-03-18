"""
Data visualization module
"""

from .maps import plot_syn, anim_syn, anim_map_with_flux
from .hists import plot_aflux, plot_hist, plot_lat

__all__ = [
    "plot_syn",
    "anim_syn",
    "anim_map_with_flux",
    "plot_aflux",
    "plot_hist",
    "plot_lat"
]
