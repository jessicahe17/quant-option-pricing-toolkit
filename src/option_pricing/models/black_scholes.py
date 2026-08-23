import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

from option_pricing.models.base import PricingModel
from option_pricing.instruments.option import Option, ExerciseStyle
from option_pricing.market.market_environment import MarketEnvironment
from option_pricing.models.results import PricingResult, Greeks


@dataclass(frozen=True)
class _BlackScholesState:
    spot: float
    strike: float
    time: float
    volatility: float
    rate: float
    dividend: float
    d1: float | None = None
    d2: float | None = None



class BlackScholesModel(PricingModel):
    """
    Black-Scholes pricing model for European options.
    Supports:
    - European calls
    - European puts
    - continuous dividend yield
    Prices are returned per underlying unit.
    """


    def evaluate(self, option: Option, market: MarketEnvironment) -> PricingResult:
        if option.exercise_style != ExerciseStyle.EUROPEAN:
            raise NotImplementedError("Black-Scholes only supports European options.")

        state = self._build_state(option, market)
        price = self._calculate_price(option, state)
        greeks = self._calculate_greeks(option, state)
        return PricingResult(price=price, greeks=greeks)


    def _calculate_greeks(self, option: Option, state: _BlackScholesState) -> Greeks:
        if (state.d1 is None) or (state.d2 is None):
            return Greeks(
                delta=None,
                gamma=None,
                vega=None,
                theta=None,
                rho=None
                )

        return Greeks(
            delta=self._calculate_delta(option, state),
            gamma=self._calculate_gamma(state),
            vega=self._calculate_vega(state),
            theta=self._calculate_theta(option, state),
            rho=self._calculate_rho(option,state)
            )


    def _calculate_price(self, option: Option, state: _BlackScholesState):
        if state.time <= 0:
            return option.payoff(state.spot)

        if state.volatility == 0:
            return self._deterministic_price(option, state)

        if option.is_call:
            return self._calculate_call_price(state)
        elif option.is_put:
            return self._calculate_put_price(state)
        else:
            raise ValueError("Unsupported option type.")
        

    def _calculate_call_price(self, state: _BlackScholesState):
        price = (
            state.spot * np.exp(-state.dividend * state.time) * norm.cdf(state.d1)
            -
            state.strike * np.exp(-state.rate * state.time) * norm.cdf(state.d2)
            )
        
        return price


    def _calculate_put_price(self, state: _BlackScholesState):
        price = (
            state.strike * np.exp(-state.rate * state.time) * norm.cdf(-state.d2)
            -
            state.spot * np.exp(-state.dividend * state.time) * norm.cdf(-state.d1)
            )

        return price


    def _calculate_d1_d2(self, S: float, K: float, T: float, sigma: float, r: float, q: float) -> tuple[float, float]:

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2


    def _calculate_time_to_expiry(self, option: Option, market: MarketEnvironment) -> float:
        days = (option.expiration_date - market.valuation_date).days
        return days / 365


    def _deterministic_price(self, option: Option, state: _BlackScholesState):
        terminal_spot = state.spot * np.exp((state.rate - state.dividend) * state.time)
        payoff = option.payoff(terminal_spot)
        return payoff * np.exp(-state.rate * state.time)


    def _build_state(self, option: Option, market: MarketEnvironment) -> _BlackScholesState:

        market_data = market.get_data(option.underlying)

        S = market_data.spot
        K = option.strike
        T = self._calculate_time_to_expiry(option, market)
        sigma = market_data.volatility
        r = market_data.risk_free_rate
        q = market_data.dividend_yield

        d1 = None
        d2 = None

        if sigma != 0 and T > 0:
            d1, d2 = self._calculate_d1_d2(S, K, T, sigma, r, q)

        return _BlackScholesState(spot=S, strike=K, time=T, volatility=sigma, rate=r, dividend=q, d1=d1, d2=d2)


    def _calculate_delta(self, option: Option, state: _BlackScholesState) -> float:
        discounted_probability = (np.exp(-state.dividend * state.time) * norm.cdf(state.d1))

        if option.is_call:
            return discounted_probability

        elif option.is_put:
            return discounted_probability - np.exp(-state.dividend * state.time)

        else:
            raise ValueError("Unsupported option type.")

    def _calculate_gamma(self, state: _BlackScholesState) -> float:
        gamma = (
            np.exp(-state.dividend * state.time) * norm.pdf(state.d1) / 
            (state.spot * state.volatility * np.sqrt(state.time))
            )

        return gamma

    def _calculate_vega(self, state: _BlackScholesState) -> float:
        vega = state.spot * np.exp(-state.dividend * state.time) * norm.pdf(state.d1) * np.sqrt(state.time)
        return vega


    def _calculate_theta(self, option: Option, state: _BlackScholesState) -> float:
        first_term = ((-state.spot * np.exp(-state.dividend * state.time) * norm.pdf(state.d1) * state.volatility)
                      / (2 * np.sqrt(state.time)))

        if option.is_call:
            theta = (first_term + state.dividend * state.spot * np.exp(-state.dividend * state.time) * norm.cdf(state.d1)
                     - state.rate * state.strike * np.exp(-state.rate * state.time) * norm.cdf(state.d2))

        elif option.is_put:
            theta = (first_term - state.dividend * state.spot * np.exp(-state.dividend * state.time) * norm.cdf(-state.d1)
                     + state.rate * state.strike * np.exp(-state.rate * state.time) * norm.cdf(-state.d2))

        else:
            raise ValueError("Unsupported option type.")

        return theta


    def _calculate_rho(self, option: Option, state: _BlackScholesState) -> float:
        if option.is_call:
            rho = state.strike * state.time * np.exp(-state.rate * state.time) * norm.cdf(state.d2)

        elif option.is_put:
            rho = -state.strike * state.time * np.exp(-state.rate * state.time) * norm.cdf(-state.d2)

        else:
            raise ValueError("Unsupported option type.")

        return rho