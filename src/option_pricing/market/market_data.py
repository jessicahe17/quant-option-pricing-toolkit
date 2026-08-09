from dataclasses import dataclass


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

        if self.spot <= 0:
            raise ValueError("Spot price must be positive.")

        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")

        if self.risk_free_rate <= -1:
            raise ValueError("Risk-free rate must be greater than -100%.")

        if self.dividend_yield < 0:
            raise ValueError("Dividend yield cannot be negative.")