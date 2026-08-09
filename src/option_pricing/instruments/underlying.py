from abc import ABC
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Underlying(ABC):
    """
    Abstract base class representing a financial underlying asset.
    Stores only identity information.
    """

    symbol: str

    def __post_init__(self):
        if not isinstance(self.symbol, str):
            raise TypeError("Symbol must be a string.")
        
        normalized_symbol = self.symbol.upper()

        if not normalized_symbol:
            raise ValueError("Symbol cannot be empty.")

        if not re.match(r"^[A-Z0-9.-]+$", normalized_symbol):
            raise ValueError("Symbol contains invalid characters.")

        object.__setattr__(self, "symbol", normalized_symbol)



@dataclass(frozen=True)
class Equity(Underlying):
    """Represents an equity security."""

    exchange: str

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.exchange, str):
            raise TypeError("Exchange must be a string.")
        
        if not self.exchange:
            raise ValueError("Exchange cannot be empty.")

        if self.exchange != self.exchange.strip():
            raise ValueError("Exchange cannot contain leading or trailing whitespace.")

        object.__setattr__(self, "exchange", self.exchange.upper())