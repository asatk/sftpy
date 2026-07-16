import abc
import numpy as np

from ...component import Component


class MagneticRegion(Component, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sample_flux(self,
                    source: float) -> np.ndarray:
        ...

    @abc.abstractmethod
    def sample_phi(self,
                   flux: np.ndarray,
                   ntotal: int) -> np.ndarray:
        ...

    @abc.abstractmethod
    def sample_theta(self,
                     flux: np.ndarray,
                     latsource: float,
                     ntotal: int) -> np.ndarray:
        ...

    @abc.abstractmethod
    def sample_orientation(self,
                           phi: np.ndarray,
                           theta: np.ndarray,
                           flux: np.ndarray,
                           source: float,
                           ntotal: int) -> np.ndarray:
        ...

    @abc.abstractmethod
    def make_concentrations(self,
                            phi: np.ndarray,
                            theta: np.ndarray,
                            flux: np.ndarray,
                            orient: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...
