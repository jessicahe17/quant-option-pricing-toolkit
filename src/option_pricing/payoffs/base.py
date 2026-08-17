from abc import ABC, abstractmethod
from typing import TypeAlias
from enum import Enum, auto

import numpy as np


Numeric: TypeAlias = float | np.ndarray
TerminalValues: TypeAlias = np.ndarray
Paths: TypeAlias = np.ndarray


class PayoffRequirement(Enum):
    """Information required by a payoff for evaluation."""

    TERMINAL = auto()
    PATH = auto()



class Payoff(ABC):
    """Abstract interface for option payoff functions."""

    @property
    @abstractmethod
    def requirement(self) -> PayoffRequirement:
        """Return the simulation information required by the payoff."""
        ...


    @abstractmethod
    def __call__(self, values: Numeric) -> Numeric:
        """Evaluate the payoff at the given underlying value."""
        ...



class TerminalPayoff(Payoff):
    """Abstract payoff that depends only on terminal underlying values."""

    @property
    def requirement(self) -> PayoffRequirement:
        """Return the information required by the payoff."""
        return PayoffRequirement.TERMINAL

    @abstractmethod
    def __call__(self, terminal_values: TerminalValues) -> Numeric:
        """Evaluate the payoff on terminal underlying values."""
        ...



class PathDependentPayoff(Payoff):
    """Abstract payoff that depends on simulated underlying paths."""

    @property
    def requirement(self) -> PayoffRequirement:
        """Return the information required by the payoff."""
        return PayoffRequirement.PATH

    @abstractmethod
    def __call__(self, paths: Paths) -> Numeric:
        """Evaluate the payoff on simulated underlying paths."""
        ...