import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from option_pricing.market.market_data import MarketData

def test_valid_market_data_creation():
    data = MarketData(
        spot=100,
        volatility=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.02
        )
    
    assert data.spot == 100
    assert data.volatility == 0.25
    assert data.risk_free_rate == 0.05
    assert data.dividend_yield == 0.02


def test_default_dividend_yield():
    data = MarketData(
            spot=100,
            volatility=0.2,
            risk_free_rate=0.05,
            )

    assert data.spot == 100
    assert data.volatility == 0.2
    assert data.risk_free_rate == 0.05
    assert data.dividend_yield == 0


def test_invalid_spot():
    with pytest.raises(ValueError, match="Spot price must be positive."):
        MarketData(spot=0, volatility=0.25, risk_free_rate=0.05)
        
    with pytest.raises(ValueError, match="Spot price must be positive."):
        MarketData(spot=-10, volatility=0.2, risk_free_rate=0.03)


def test_invalid_volatility():
    with pytest.raises(ValueError, match="Volatility cannot be negative."):
        MarketData(spot=120, volatility=-0.1, risk_free_rate=0.05)


def test_negative_but_valid_interest_rate():
    data = MarketData(spot=100, volatility=0.2, risk_free_rate=-0.05)
    assert data.risk_free_rate == -0.05



def test_invalid_dividend_yield():
    with pytest.raises(ValueError, match="Dividend yield cannot be negative."):
        MarketData(spot=110, volatility=0.1, risk_free_rate=0.05, dividend_yield=-0.01)


def test_market_data_immutability():
    data = MarketData(spot=100, volatility=0.25, risk_free_rate=0.05)
    
    with pytest.raises(FrozenInstanceError):
        data.spot = 105
        
    with pytest.raises(FrozenInstanceError):
        data.volatility = 0.3

    with pytest.raises(FrozenInstanceError):
        data.risk_free_rate = 0.1

    with pytest.raises(FrozenInstanceError):
        data.dividend_yield = 0.02


def test_large_negative_interest_rate_is_allowed():
    data = MarketData(
        spot=100,
        volatility=0.2,
        risk_free_rate=-1.05,
    )

    assert data.risk_free_rate == -1.05


def test_market_data_rejects_boolean_values():
    with pytest.raises(TypeError, match="spot must be a numeric value."):
        MarketData(
            spot=True,
            volatility=0.2,
            risk_free_rate=0.05,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spot", np.nan),
        ("volatility", np.inf),
        ("risk_free_rate", -np.inf),
        ("dividend_yield", np.nan),
    ],
)
def test_market_data_rejects_non_finite_values(field, value):
    kwargs = {
        "spot": 100.0,
        "volatility": 0.2,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.02,
    }

    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a finite number."):
        MarketData(**kwargs)