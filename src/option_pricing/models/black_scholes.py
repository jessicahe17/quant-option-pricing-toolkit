import numpy as np
from scipy.stats import norm

from option_pricing.models.base import PricingModel
from option_pricing.instruments.option import Option, ExerciseStyle
from option_pricing.market.market_environment import MarketEnvironment


class BlackScholesModel(PricingModel):
    """
    Black-Scholes pricing model for European options.
    Supports:
    - European calls
    - European puts
    - continuous dividend yield
    Prices are returned per underlying unit.
    """


    def price(self, option: Option, market: MarketEnvironment) -> float:

        if option.exercise_style != ExerciseStyle.EUROPEAN:
            raise NotImplementedError("Black-Scholes only supports European options.")

        S, K, T, sigma, r, q = self._extract_parameters(option, market)

        if T <= 0:
            return option.payoff(S)

        if sigma == 0:
            return self._deterministic_price(option, S, K, T, r, q)

        if option.is_call:
            return self.price_call(S, K, T, sigma, r, q)
        elif option.is_put:
            return self.price_put(S, K, T, sigma, r, q)
        else:
            raise ValueError("Unsupported option type.")


    def price_call(self, S: float, K: float, T: float, sigma: float, r: float, q: float) -> float:
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma, r, q)
        price = (
            S * np.exp(-q * T) * norm.cdf(d1)
            -
            K * np.exp(-r * T) * norm.cdf(d2)
        )

        return price


    def price_put(self, S: float, K: float, T: float, sigma: float, r: float, q: float) -> float:
        d1, d2 = self._calculate_d1_d2(S, K, T, sigma, r, q)
        price = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            -
            S * np.exp(-q * T) * norm.cdf(-d1)
        )

        return price


    def _calculate_d1_d2(self, S: float, K: float, T: float, sigma: float, r: float, q: float) -> tuple[float, float]:

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2


    def _extract_parameters(self, option: Option, market: MarketEnvironment) -> tuple[float, float, float, float, float, float]:

        market_data = market.get_data(option.underlying)

        S = market_data.spot
        K = option.strike
        T = self._calculate_time_to_expiry(option, market)
        sigma = market_data.volatility
        r = market_data.risk_free_rate
        q = market_data.dividend_yield

        return S, K, T, sigma, r, q


    def _calculate_time_to_expiry(self, option: Option, market: MarketEnvironment) -> float:
        days = (option.expiration_date - market.valuation_date).days
        return days / 365


    def _deterministic_price(self, option: Option, S: float, K: float, T: float, r: float, q: float) -> float:
        terminal_spot = S * np.exp((r - q) * T)
        payoff = option.payoff(terminal_spot)
        return payoff * np.exp(-r * T)