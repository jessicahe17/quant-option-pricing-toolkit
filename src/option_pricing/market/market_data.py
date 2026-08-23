from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MarketData:
    """
    A snapshot of the market state for a single underlying asset at a specific time.
    Strictly a data container; contains no pricing logic or calculations.
    """

    spot: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float = 0.0


    def __post_init__(self):
        fields = {
            "spot": self.spot,
            "volatility": self.volatility,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
        }

        for name, value in fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a numeric value.")

            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if self.spot <= 0:
            raise ValueError("Spot price must be positive.")

        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")

        if self.dividend_yield < 0:
            raise ValueError("Dividend yield cannot be negative.")