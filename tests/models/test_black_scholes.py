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


@pytest.fixture
def call_opt(underlying):
    return Option(
        underlying=underlying, 
        strike=100,
        expiration_date=date(2027,1,1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN
        )


@pytest.fixture
def put_opt(underlying):
    return Option(
        underlying=underlying, 
        strike=100,
        expiration_date=date(2027,1,1),
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN
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
    price = model.evaluate(option, market).price

    assert price == pytest.approx(10.45, abs=1e-3)


def test_european_put_price(underlying, market):
    option = Option(
        underlying=underlying,
        strike=100,
        expiration_date=date(2027,1,1),
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN
        )

    price = BlackScholesModel().evaluate(option, market).price

    assert price == pytest.approx(5.573, abs=1e-3)


def test_put_call_parity(underlying, market):
    call = Option(underlying, 100, date(2027,1,1), OptionType.CALL, ExerciseStyle.EUROPEAN)
    put = Option(underlying, 100, date(2027,1,1), OptionType.PUT, ExerciseStyle.EUROPEAN)

    model = BlackScholesModel()
    call_price = model.evaluate(call, market).price
    put_price = model.evaluate(put, market).price

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
        BlackScholesModel().evaluate(option, market).price


def test_boundary_depp_otm_call(underlying):
    call = Option(underlying, 100, date(2027,1,1), OptionType.CALL, ExerciseStyle.EUROPEAN)
    market = MarketEnvironment({underlying: MarketData(1, 0.2, 0.05)}, date(2026, 1, 1))
    model = BlackScholesModel()

    price = model.evaluate(call, market).price
    assert price < 1e-6


def test_boundary_depp_otm_put(underlying):
    put = Option(underlying, 10, date(2027,1,1), OptionType.PUT, ExerciseStyle.EUROPEAN)
    market = MarketEnvironment({underlying: MarketData(10000, 0.2, 0.05)}, date(2026, 1, 1))
    model = BlackScholesModel()

    price = model.evaluate(put, market).price
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

    price_without_dividend = model.evaluate(option, market_without_dividend).price
    price_with_dividend = model.evaluate(option, market_with_dividend).price

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

    price = BlackScholesModel().evaluate(option, market).price

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
    price = model.evaluate(option, market).price
    terminal_spot = (100 * np.exp(0.05 * 1))
    expected = max(terminal_spot - 100, 0) * np.exp(-0.05)

    assert price == pytest.approx(expected)


def test_call_delta(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)

    assert result.greeks.delta == pytest.approx(0.6368, rel=1e-4)


def test_put_delta(put_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(put_opt, market)

    assert result.greeks.delta == pytest.approx(-0.3632, rel=1e-4)


def test_put_call_delta_parity(call_opt, put_opt, market):
    model = BlackScholesModel()
    call_delta = model.evaluate(call_opt, market).greeks.delta
    put_delta = model.evaluate(put_opt, market).greeks.delta

    assert call_delta - put_delta == pytest.approx(1, rel=1e-4)


def test_gamma_positive(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)
    assert result.greeks.gamma > 0


def test_put_call_gamma_equal(call_opt, put_opt, market):
    model = BlackScholesModel()
    call_result = model.evaluate(call_opt, market)
    put_result = model.evaluate(put_opt, market)
    assert call_result.greeks.gamma == pytest.approx(put_result.greeks.gamma)


def test_gamma_analytical_value(call_opt, market):
    model = BlackScholesModel()
    assert model.evaluate(call_opt, market).greeks.gamma == pytest.approx(0.01876, rel=1e-3)


def test_vega_positive(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)
    assert result.greeks.vega > 0


def test_put_call_vega_equal(call_opt, put_opt, market):
    model = BlackScholesModel()
    call_result = model.evaluate(call_opt, market)
    put_result = model.evaluate(put_opt, market)
    assert call_result.greeks.vega == pytest.approx(put_result.greeks.vega)


def test_vega_analytical_value(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)
    assert result.greeks.vega == pytest.approx(37.524, rel=1e-3)


def test_negative_theta(call_opt, put_opt, market):
    model = BlackScholesModel()
    call_result = model.evaluate(call_opt, market)
    put_result = model.evaluate(put_opt,market)
    assert call_result.greeks.theta < 0
    assert put_result.greeks.theta < 0


def test_call_put_theta_differ(call_opt, put_opt, market):
    model = BlackScholesModel()
    call_result = model.evaluate(call_opt, market)
    put_result = model.evaluate(put_opt, market)
    assert call_result.greeks.theta != put_result.greeks.theta 


def test_theta_analytical_value(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)
    assert result.greeks.theta == pytest.approx(-1.0908, rel=1e-3)


def test_call_positive_rho(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)
    assert result.greeks.rho > 0


def test_put_negative_rho(put_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(put_opt, market)
    assert result.greeks.rho < 0


def test_rho_analytical_value(call_opt, market):
    model = BlackScholesModel()
    result = model.evaluate(call_opt, market)
    assert result.greeks.rho == pytest.approx(53.232, rel=1e-3)