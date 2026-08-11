from abc import ABC, abstractmethod

from option_pricing.instruments.option import Option
from option_pricing.market.market_environment import MarketEnvironment
from option_pricing.models.results import PricingResult

class PricingModel(ABC):
    """
    Abstract base class for all option pricing models.

    Defines the standard interface that any concrete pricing model must implement.
    """

@abstractmethod
def evaluate(self, option: Option, market: MarketEnvironment) -> PricingResult:
    """
    Calculate the theoretical price of an option.

    Parameters
    ----------
    option : Option
        The option contract being priced.

    market : MarketEnvironment
        The market environment containing relevant market data 

    Returns
    -------
    PricingResult:
        Theoretical option value and associated risk sensitivities.
    """

    pass