import pytest
import numpy as np
from datetime import date

from option_pricing.instruments.option import Option, OptionType, ExerciseStyle
from option_pricing.instruments.underlying import Equity
from option_pricing.market.market_data import MarketData
from option_pricing.market.market_environment import MarketEnvironment
from option_pricing.models.black_scholes import BlackScholesModel

@pytest.fixture
def underlying():
    return Equity("AAPL", "NASDAQ")


@pytest.fixture
def market(underlying):
    return MarketEnvironment(
        {underlying: MarketData(
            spot=100,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0
            )
            },
        date(2026, 1, 1)
        )


def test_european_call_price(underlying, market):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2027,1,1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN
        )

    model = BlackScholesModel()
    price = model.price(option, market)

    assert price == pytest.approx(10.45, abs=1e-3)


def test_european_put_price(underlying, market):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2027,1,1),
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN
        )

    price = BlackScholesModel().price(option, market)

    assert price == pytest.approx(5.573, abs=1e-3)


def test_put_call_parity(underlying, market):
    call = Option(underlying, 100, date(2027,1,1), OptionType.CALL, ExerciseStyle.EUROPEAN)
    put = Option(underlying, 100, date(2027,1,1), OptionType.PUT, ExerciseStyle.EUROPEAN)

    model = BlackScholesModel()
    call_price = model.price(call, market)
    put_price = model.price(put, market)

    lhs = call_price - put_price
    rhs = (market.get_data(underlying).spot - 
           call.strike * np.exp(- market.get_data(underlying).risk_free_rate * 1))

    assert lhs == pytest.approx(rhs) 


def test_black_scholes_rejects_american_option(underlying, market):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2027,1,1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN
        )

    with pytest.raises(NotImplementedError, match="Black-Scholes only supports European options."):
        BlackScholesModel().price(option, market)


def test_boundary_depp_otm_call(underlying):
    call = Option(underlying, 100, date(2027,1,1), OptionType.CALL, ExerciseStyle.EUROPEAN)
    market = MarketEnvironment({underlying: MarketData(1, 0.2, 0.05)}, date(2026, 1, 1))
    model = BlackScholesModel()

    price = model.price(call, market)
    assert price < 1e-6


def test_boundary_depp_otm_put(underlying):
    put = Option(underlying, 10, date(2027,1,1), OptionType.PUT, ExerciseStyle.EUROPEAN)
    market = MarketEnvironment({underlying: MarketData(10000, 0.2, 0.05)}, date(2026, 1, 1))
    model = BlackScholesModel()

    price = model.price(put, market)
    assert price < 1e-6


def test_dividend_yield_reduces_call_value(underlying):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2027, 1, 1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN
        )

    market_without_dividend = MarketEnvironment(
        {
            underlying: MarketData(
            spot=100,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0
            )
            },
        date(2026, 1, 1)
        )

    market_with_dividend = MarketEnvironment(
        {
            underlying: MarketData(
            spot=100,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0.03
            )
            },
        date(2026, 1, 1)
        )

    model = BlackScholesModel()

    price_without_dividend = model.price(option, market_without_dividend)
    price_with_dividend = model.price(option, market_with_dividend)

    assert price_with_dividend < price_without_dividend


def test_expired_call_returns_payoff(underlying):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2026, 1, 1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN
        )


    market = MarketEnvironment(
        {
            underlying: MarketData(
            spot=120,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0
            )
            },
        date(2026, 1, 1)
        )

    price = BlackScholesModel().price(option, market)

    assert price == pytest.approx(20)


def test_zero_volatility_price(underlying):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2027, 1, 1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN
        )


    market = MarketEnvironment(
        {
            underlying: MarketData(
            spot=100,
            volatility=0,
            risk_free_rate=0.05,
            dividend_yield=0
            )
            },
        date(2026, 1, 1)
        )


    model = BlackScholesModel()
    price = model.price(option, market)
    terminal_spot = (100 * np.exp(0.05 * 1))
    expected = max(terminal_spot - 100, 0) * np.exp(-0.05)

    assert price == pytest.approx(expected)