import numpy as np
from scipy.stats import norm

from option_pricing.models.base import PricingModel
from option_pricing.instruments.option import Option, OptionType, ExerciseStyle
from option_pricing.market.market_environment import MarketEnvironment


class BlackScholesModel(PricingModel):
    """Black-Scholes pricing model for European options."""


    def price(self, option: Option, market: MarketEnvironment) -> float:

        if option.exercise_style != ExerciseStyle.EUROPEAN:
            raise NotImplementedError("Black-Scholes only supports European options.")

        if option.option_type == OptionType.CALL:
            return self.price_call(option, market)
        elif option.option_type == OptionType.PUT:
            return self.price_put(option, market)
        else:
            raise ValueError("Unsupported option type.")


    def price_call(self, option: Option, market: MarketEnvironment) -> float:

        S, K, T, sigma, r = self._extract_parameters(option, market)
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma, r)
        price = (
            S * norm.cdf(d1)
            -
            K * np.exp(-r * T) * norm.cdf(d2)
        )

        return price


    def price_put(self, option: Option, market: MarketEnvironment) -> float:

        S, K, T, sigma, r = self._extract_parameters(option, market)
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma, r)
        price = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            -
            S * norm.cdf(-d1)
        )

        return price


    def _calculate_d1_d2(self, S: float, K: float, T: float, sigma: float, r: float) -> tuple[float, float]:

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2


    def _extract_parameters(self, option: Option, market: MarketEnvironment):

        market_data = market.get_data(option.underlying)

        S = market_data.spot
        K = option.strike
        T = (option.expiration_date - market.valuation_date).days / 365
        sigma = market_data.volatility
        r = market_data.risk_free_rate

        return S, K, T, sigma, r