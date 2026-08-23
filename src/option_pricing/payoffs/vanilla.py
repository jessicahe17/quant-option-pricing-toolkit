import numpy as np
from dataclasses import dataclass
from option_pricing.payoffs.base import Numeric, TerminalPayoff, PathDependentPayoff


@dataclass(frozen=True)
class EuropeanCall(TerminalPayoff):
    """Payoff of a European call option."""

    strike: float

    def __post_init__(self) -> None:
        if isinstance(self.strike, (bool, np.bool_)) or not isinstance(
            self.strike,
            (int, float, np.integer, np.floating)
            ):
            raise TypeError("strike must be a numeric value.")

        if not np.isfinite(self.strike):
            raise ValueError("strike must be a finite number.")

        if self.strike <= 0:
            raise ValueError("strike must be strictly positive.")


    def __call__(self, underlying_value: Numeric) -> Numeric:
        return np.maximum(underlying_value - self.strike, 0.0)



@dataclass(frozen=True)
class EuropeanPut(TerminalPayoff):
    """Payoff of a European put option."""

    strike: float

    def __post_init__(self) -> None:
        if isinstance(self.strike, (bool, np.bool_)) or not isinstance(
            self.strike,
            (int, float, np.integer, np.floating)
            ):
            raise TypeError("strike must be a numeric value.")

        if not np.isfinite(self.strike):
            raise ValueError("strike must be a finite number.")

        if self.strike <= 0:
            raise ValueError("strike must be strictly positive.")


    def __call__(self, underlying_value: Numeric) -> Numeric:
        return np.maximum(self.strike - underlying_value, 0.0)



@dataclass(frozen=True)
class AsianCall(PathDependentPayoff):
    """Payoff of an arithmetic-average Asian call option."""

    strike: float

    def __post_init__(self) -> None:
        if isinstance(self.strike, (bool, np.bool_)) or not isinstance(
            self.strike,
            (int, float, np.integer, np.floating)
            ):
            raise TypeError("strike must be a numeric value.")

        if not np.isfinite(self.strike):
            raise ValueError("strike must be a finite number.")

        if self.strike <= 0:
            raise ValueError("strike must be strictly positive.")


    def __call__(self, paths: np.ndarray) -> Numeric:
        if not isinstance(paths, np.ndarray):
            raise TypeError("paths must be a NumPy array.")

        if paths.ndim != 2:
            raise ValueError("paths must be 2-dimensional.")

        if paths.shape[0] < 1:
            raise ValueError("paths must contain at least one path.")

        if paths.shape[1] < 2:
            raise ValueError("paths must contain at least two time points.")

        if not np.issubdtype(paths.dtype, np.number):
            raise TypeError("paths must have a numeric dtype.")

        if not np.all(np.isfinite(paths)):
            raise ValueError("paths must contain only finite values.")

        average = np.mean(paths[:, 1:], axis=1)

        return np.maximum(average - self.strike, 0.0)