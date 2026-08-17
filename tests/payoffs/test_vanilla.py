import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from option_pricing.payoffs.vanilla import EuropeanCall, EuropeanPut, AsianCall
from option_pricing.payoffs.base import PayoffRequirement


def test_valid_creation():
    call = EuropeanCall(strike=100.0)
    put = EuropeanPut(strike=100.0)
    asian_call = AsianCall(strike=100.0)

    assert call.strike == 100.0
    assert put.strike == 100.0
    assert asian_call.strike == 100.0


def test_invalid_strike_type():
    with pytest.raises(TypeError, match="strike must be a numeric value."):
        EuropeanCall("100")

    with pytest.raises(TypeError, match="strike must be a numeric value."):
        EuropeanPut("100")

    with pytest.raises(TypeError, match="strike must be a numeric value."):
        AsianCall("100")


def test_invalid_strike_value():
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanCall(strike=0)
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanCall(strike=-1)

    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanPut(strike=0)
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        EuropeanPut(strike=-1)

    with pytest.raises(ValueError, match="strike must be strictly positive."):
        AsianCall(strike=0)
    with pytest.raises(ValueError, match="strike must be strictly positive."):
        AsianCall(strike=-1)


def test_european_call_payoff_values():
    underlying = np.array([90.0, 100.0, 110.0])
    expected = np.array([0.0, 0.0, 10.0])
    call = EuropeanCall(strike=100)

    np.testing.assert_allclose(call(underlying), expected)


def test_european_put_payoff_values():
    underlying = np.array([90.0, 100.0, 110.0])
    expected = np.array([10.0, 0.0, 0.0])
    put = EuropeanPut(strike=100)

    np.testing.assert_allclose(put(underlying), expected)


def test_immutability():
    call = EuropeanCall(strike=100)
    put = EuropeanPut(strike=100)
    asian_call = AsianCall(strike=100)

    with pytest.raises(FrozenInstanceError):
        call.strike = 90.0

    with pytest.raises(FrozenInstanceError):
        put.strike = 110.0

    with pytest.raises(FrozenInstanceError):
        asian_call.strike = 90.0


def test_asian_call_basic_arithmetic_average():
    asian_call = AsianCall(strike=105.0)
    paths = np.array([[100.0, 110.0, 120.0]])
    expected = np.array([10.0])
    result = asian_call(paths)

    np.testing.assert_allclose(result, expected)


def test_asian_call_out_of_the_money():
    asian_call = AsianCall(strike=110.0)
    paths = np.array([[100.0, 101.0, 102.0]])
    expected = np.array([0.0])
    result = asian_call(paths)

    np.testing.assert_allclose(result, expected)


def test_asian_call_vectorization():
    asian_call = AsianCall(strike=100.0)
    paths = np.array([
        [100.0, 110.0, 120.0],
        [100.0,  90.0,  80.0],
        [100.0, 105.0, 115.0]
        ])
    expected = np.array([15.0, 0.0, 10.0])
    result = asian_call(paths)

    np.testing.assert_allclose(result, expected)


def test_asian_call_requirement():
    asian_call = AsianCall(strike=100.0)
    assert asian_call.requirement == PayoffRequirement.PATH 


def test_asian_call_invalid_paths():
    asian_call = AsianCall(strike=100.0)
    
    with pytest.raises(TypeError, match="paths must be a NumPy array."):
        asian_call([[100.0, 110.0, 120.0]])
        
    with pytest.raises(ValueError, match="paths must be 2-dimensional."):
        asian_call(np.array([100.0, 110.0, 120.0]))

    with pytest.raises(ValueError, match="paths must contain at least one path."):
        asian_call(np.empty((0, 5)))
        
    with pytest.raises(ValueError, match="paths must contain at least two time points."):
        asian_call(np.array([[100.0], [110.0]]))
        
    with pytest.raises(TypeError, match="paths must have a numeric dtype."):
        asian_call(np.array([["100.0", "110.0", "120.0"]]))
        
    with pytest.raises(ValueError, match="paths must contain only finite values."):
        asian_call(np.array([[100.0, np.nan, 120.0]]))
        
    with pytest.raises(ValueError, match="paths must contain only finite values."):
        asian_call(np.array([[100.0, 110.0, np.inf]]))