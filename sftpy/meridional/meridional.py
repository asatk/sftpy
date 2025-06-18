"""
Meridional flow
"""

import numpy as np


def meridional(theta: np.ndarray, dt: float, amplifier: float, mode: int=1):
    """
    Only does profile 1 (Komm et al. 93)
    """
    return meridional_1(theta, dt, amplifier)


def meridional_1(theta: np.ndarray, dt: float, amplifier: float):
    """
    Move meridional flow. Assumes theta is nflux elements.
    """
    scale = dt / 7e5
    a = 12.9 - 3 * scale * amplifier
    b = 1.4 - 3 * scale * amplifier
    return theta - a * np.sin(2*theta) + b * np.sin(4*theta)
