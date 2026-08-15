import pytest
import numpy as np

from option_pricing.payoffs.vanilla import EuropeanPut, EuropeanCall
from option_pricing.simulation.results import MonteCarloPricingResult
from option_pricing.simulation.monte_carlo import MonteCarloPricer


@pytest.fixture
def call():
    return EuropeanCall(strike=100.0)


@pytest.fixture
def put():
    return EuropeanPut(strike=100.0)


@pytest.fixture
def pricer(call):
    return MonteCarloPricer(payoff=call, rate=0.0, maturity=1.0)


def test_valid_construction(call):
    pricer = MonteCarloPricer(payoff=call, rate=0.05, maturity=1.0)

    assert pricer.payoff is call
    assert pricer.rate == 0.05
    assert pricer.maturity == 1.0


def test_invalid_payoff():
    with pytest.raises(TypeError, match="payoff must be a Payoff."):
        MonteCarloPricer(payoff="Not a payoff", rate=0.05, maturity=1.0)


def test_invalid_rate(call):
    with pytest.raises(TypeError, match="rate must be a numeric value."):
        MonteCarloPricer(payoff=call, rate="0.05", maturity=1.0)

    with pytest.raises(ValueError, match="rate must be a finite number."):
        MonteCarloPricer(payoff=call, rate=np.inf, maturity=1.0)


def test_invalid_maturity(call):
    with pytest.raises(TypeError, match="maturity must be a numeric value."):
        MonteCarloPricer(payoff=call, rate=0.05, maturity="1.0")

    with pytest.raises(ValueError, match="maturity must be a finite number."):
        MonteCarloPricer(payoff=call, rate=0.05, maturity=np.inf)


def test_invalid_input_type(pricer):
    with pytest.raises(TypeError, match="terminal_values must be a NumPy array."):
        pricer.price([90.0, 100.0, 110.0])


def test_non_1d_array(pricer):
    with pytest.raises(ValueError, match="terminal_values must be a 1-D array."):
        pricer.price(np.array([[100.0, 110.0], [90.0, 105.0]]))


def test_fewer_than_2_terminal_values(pricer):
    with pytest.raises(
        ValueError, match="terminal_values must contain at least two values."
    ):
        pricer.price(np.array([100.0]))


def test_non_numeric_values(pricer):
    with pytest.raises(TypeError, match="terminal_values must contain numeric values."):
        pricer.price(np.array(["100.0", "110.0"]))


def test_non_finite_values(pricer):
    with pytest.raises(
        ValueError, match="terminal_values must contain only finite values."
    ):
        pricer.price(np.array([100.0, np.inf]))


def test_correct_call_price_small_sample(pricer):
    terminal_values = np.array([90.0, 110.0, 130.0])
    result = pricer.price(terminal_values)

    assert isinstance(result, MonteCarloPricingResult)
    assert np.isclose(result.price, 40.0 / 3.0)


def test_correct_put_price_small_sample(put):
    pricer = MonteCarloPricer(payoff=put, rate=0.0, maturity=1.0)
    terminal_values = np.array([90.0, 100.0, 120.0])
    result = pricer.price(terminal_values)

    assert np.isclose(result.price, 10.0 / 3.0)


def test_correct_n_paths(pricer):
    terminal_values = np.array([90.0, 100.0, 110.0, 120.0, 130.0])
    result = pricer.price(terminal_values)

    assert result.n_paths == 5


def test_correct_standard_error(pricer):
    terminal_values = np.array([100.0, 110.0, 120.0])
    expected_se = 10.0 / np.sqrt(3)
    result = pricer.price(terminal_values)

    assert np.isclose(result.standard_error, expected_se)


def test_correct_discounting(call):
    rate = 0.05
    maturity = 2.0
    discount_factor = np.exp(-rate * maturity)

    pricer = MonteCarloPricer(payoff=call, rate=rate, maturity=maturity)
    terminal_values = np.array([100.0, 110.0, 120.0])

    expected_price = discount_factor * 10.0
    expected_se = discount_factor * 10.0 / np.sqrt(3)

    result = pricer.price(terminal_values)

    assert np.isclose(result.price, expected_price)
    assert np.isclose(result.standard_error, expected_se)
