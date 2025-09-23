import abc
import numpy as np

from sftpy import simrc as rc

from ..component import Component

loglvl = rc["component.loglvl"]
nfluxmax = rc["general.nfluxmax"]

class Initialize(Component, metaclass=abc.ABCMeta):

    def __init__(self,
                 nfluxmax: int=nfluxmax,
                 loglvl: int=loglvl):
        """
        Create an `Initialize` component, providing the maximum number of spots.

        Parameters
        ----------
        nfluxmax : int, optional
            Maximum number of spots.
        loglvl
            Maximum logging level.
        """
        super().__init__(loglvl)
        self._nfluxmax = nfluxmax

    @abc.abstractmethod
    def init(self):
        """
        Create the arrays for spots on the stellar surface.

        Returns
        -------
        phi : np.ndarray
            Longitudes of spots.
        theta : np.ndarray
            Co-latitudes of spots.
        flux : np.ndarray
            Fluxes of spots.
        nflux : np.ndarray
            Number of spots.
        """
