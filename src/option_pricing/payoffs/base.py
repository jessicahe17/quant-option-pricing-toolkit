import numpy as np

from abc import ABC, abstractmethod
from typing import TypeAlias


Numeric: TypeAlias = float | np.ndarray

class Payoff(ABC):
    """Abstract interface for option payoff functions."""

    @abstractmethod
    def __call__(self, underlying_value: Numeric) -> Numeric:
        """Evaluate the payoff at the given underlying value."""
        ...