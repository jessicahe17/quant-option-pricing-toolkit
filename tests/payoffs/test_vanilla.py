import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from option_pricing.payoffs.vanilla import EuropeanCall, EuropeanPut


def test_valid_creation():
    call = EuropeanCall(strike=100.0)
    put = EuropeanPut(strike=100.0)

    assert call.strike == 100.0
    assert put.strike == 100.0


def test_invalid_strike_type():
    with pytest.raises(TypeError, match="strike must be a numeric value."):
        EuropeanCall("100")

    with pytest.raises(TypeError, match="strike must be a numeric value."):
        EuropeanPut("100")


def test_invalid_strike_value():
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanCall(strike=0)
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanCall(strike=-1)

    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanPut(strike=0)
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanPut(strike=-1)


def test_call_payoff_values():
    underlying = np.array([90.0, 100.0, 110.0])
    expected = np.array([0.0, 0.0, 10.0])
    call = EuropeanCall(strike=100)

    np.testing.assert_allclose(call(underlying), expected)


def test_put_payoff_values():
    underlying = np.array([90.0, 100.0, 110.0])
    expected = np.array([10.0, 0.0, 0.0])
    put = EuropeanPut(strike=100)

    np.testing.assert_allclose(put(underlying), expected)


def test_immutability():
    call = EuropeanCall(strike=100)
    put = EuropeanPut(strike=100)

    with pytest.raises(FrozenInstanceError):
        call.strike = 90.0

    with pytest.raises(FrozenInstanceError):
        put.strike = 110.0