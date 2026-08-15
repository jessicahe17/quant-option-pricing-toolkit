import numpy as np
from dataclasses import dataclass
from option_pricing.payoffs.base import Payoff, Numeric


@dataclass(frozen=True)
class EuropeanCall(Payoff):
    """Payoff of a European call option."""

    strike: float

    def __post_init__(self) -> None:
        if not isinstance(self.strike, (int, float, np.integer, np.floating)):
            raise TypeError("strike must be a numeric value.")

        if self.strike <= 0:
            raise ValueError("strike must be strictly positive.")


    def __call__(self, underlying_value: Numeric) -> Numeric:
        return np.maximum(underlying_value - self.strike, 0.0)



@dataclass(frozen=True)
class EuropeanPut(Payoff):
    """Payoff of a European put option."""

    strike: float

    def __post_init__(self) -> None:
        if not isinstance(self.strike, (int, float, np.integer, np.floating)):
            raise TypeError("strike must be a numeric value.")

        if self.strike <= 0:
            raise ValueError("strike must be strictly positive.")


    def __call__(self, underlying_value: Numeric) -> Numeric:
        return np.maximum(self.strike - underlying_value, 0.0)