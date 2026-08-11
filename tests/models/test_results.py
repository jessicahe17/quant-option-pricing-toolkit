import pytest
from dataclasses import FrozenInstanceError
from option_pricing.models.results import Greeks, PricingResult


def test_greeks_creation():
    greeks = Greeks(delta=0.55, gamma=0.04, vega=0.12, theta=-0.02, rho=0.05)
    
    assert greeks.delta == 0.55
    assert greeks.gamma == 0.04
    assert greeks.vega == 0.12
    assert greeks.theta == -0.02
    assert greeks.rho == 0.05


def test_pricing_result_nesting():
    greeks = Greeks(delta=0.55, gamma=0.04, vega=0.12, theta=-0.02, rho=0.05)
    result = PricingResult(price=15.5, greeks=greeks)
    
    assert result.price == 15.5

    assert result.greeks.delta == 0.55
    assert result.greeks.gamma == 0.04
    assert result.greeks.vega == 0.12


def test_immutability():
    greeks = Greeks(delta=0.55, gamma=0.04, vega=0.12, theta=-0.02, rho=0.05)
    result = PricingResult(price=15.5, greeks=greeks)

    with pytest.raises(FrozenInstanceError):
        result.price = 10
        
    with pytest.raises(FrozenInstanceError):
        result.greeks.delta = 0.7