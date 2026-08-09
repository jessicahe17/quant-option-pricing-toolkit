from dataclasses import dataclass
from typing import Mapping
from types import MappingProxyType

from option_pricing.instruments.underlying import Underlying
from option_pricing.market.market_data import MarketData


@dataclass(frozen=True)
class MarketEnvironment:
    """
    Immutable collection of market data.
    Maps financial underlyings to their market state.
    """

    data: Mapping[Underlying, MarketData]


    def __post_init__(self):

        if not isinstance(self.data, Mapping):
            raise TypeError("Data must be provided as a Mapping.")

        if not self.data:
            raise ValueError("Market environment cannot be empty.")

        for underlying, market_data in self.data.items():
            if not isinstance(underlying, Underlying):
                raise TypeError("Keys must be Underlying objects.")
            if not isinstance(market_data, MarketData):
                raise TypeError("Values must be MarketData objects.")

        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


    def get_data(self, underlying: Underlying) -> MarketData:
        try:
            return self.data[underlying]
        except KeyError:
            raise KeyError(f"No market data found for {underlying.symbol}.")