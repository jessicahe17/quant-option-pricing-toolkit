from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np


Numeric: TypeAlias = float | np.ndarray

class StochasticProcess(ABC):
    """
    Defines the numerical stepping logic for a stochastic process.
    """

    @property
    @abstractmethod
    def initial_value(self):
        """
        Return the initial state of the process.
        """

        ...


    @abstractmethod
    def step(self, current_value: Numeric, dt: float, random_draw: Numeric) -> Numeric:
        """
        Advance the process by one time step.
        """

        ...