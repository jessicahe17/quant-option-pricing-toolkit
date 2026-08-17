import pytest
import numpy as np

from option_pricing.payoffs.vanilla import EuropeanPut, EuropeanCall, AsianCall
from option_pricing.simulation.results import MonteCarloPricingResult, SimulationResult
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
    invalid_input = np.array([100, 110])
    with pytest.raises(TypeError, match="simulation_result must be a SimulationResult."):
        pricer.price(invalid_input)


def test_correct_call_price_small_sample(pricer):
    s_result = SimulationResult(
        np.array([[100.0, 95.0, 90.0], 
                  [100.0, 105.0, 110.0], 
                  [100.0, 110.0, 130.0]]), 
        np.array([0, 0.5, 1]))
    result = pricer.price(s_result)

    assert isinstance(result, MonteCarloPricingResult)
    assert np.isclose(result.price, 40.0 / 3.0)


def test_correct_put_price_small_sample(put):
    pricer = MonteCarloPricer(payoff=put, rate=0.0, maturity=1.0)
    s_result = SimulationResult(
        np.array([[100, 95, 90], 
                  [100, 105, 100], 
                  [100, 110, 120]]),
        np.array([0, 0.5, 1]))
    result = pricer.price(s_result)

    assert np.isclose(result.price, 10.0 / 3.0)


def test_correct_n_paths(pricer):
    s_result = SimulationResult(
        np.array([[100, 95, 90], 
                  [100, 105, 100], 
                  [100, 110, 120]]),
        np.array([0, 0.5, 1]))
    result = pricer.price(s_result)

    assert result.n_paths == 3


def test_correct_standard_error(pricer):
    s_result = SimulationResult(
        np.array([[100, 105, 100], 
                  [100, 106, 110], 
                  [100, 110, 120]]),
        np.array([0, 0.5, 1]))
    expected_se = 10.0 / np.sqrt(3)
    result = pricer.price(s_result)

    assert np.isclose(result.standard_error, expected_se)


def test_correct_discounting(call):
    rate = 0.05
    maturity = 2.0
    discount_factor = np.exp(-rate * maturity)

    pricer = MonteCarloPricer(payoff=call, rate=rate, maturity=maturity)
    s_result = SimulationResult(
        np.array([[100, 105, 100], 
                  [100, 106, 110], 
                  [100, 110, 120]]),
        np.array([0, 0.5, 1]))

    expected_price = discount_factor * 10.0
    expected_se = discount_factor * 10.0 / np.sqrt(3)

    result = pricer.price(s_result)

    assert np.isclose(result.price, expected_price)
    assert np.isclose(result.standard_error, expected_se)


def test_asian_call_path_dependent_pricing():
    paths = np.array([
        [100.0, 110.0, 130.0], #120
        [100.0,  90.0,  80.0], #95
        [100.0, 105.0, 125.0], #115
    ])
    time_grid = np.array([0.0, 0.5, 1.0])
    
    simulation_result = SimulationResult(paths=paths, time_grid=time_grid)

    rate = 0.05
    maturity = 1.0
    strike = 100.0

    asian_payoff = AsianCall(strike=strike)
    pricer = MonteCarloPricer(payoff=asian_payoff, rate=rate, maturity=maturity)

    result = pricer.price(simulation_result)

    expected_payoffs = np.array([20.0, 0.0, 15.0])
    discount_factor = np.exp(-rate * maturity)
    
    expected_price = discount_factor * np.mean(expected_payoffs)
    expected_std = np.std(expected_payoffs, ddof=1)
    expected_stderr = discount_factor * expected_std / np.sqrt(3)

    assert result.n_paths == 3
    assert np.isclose(result.price, expected_price)
    assert np.isclose(result.standard_error, expected_stderr)