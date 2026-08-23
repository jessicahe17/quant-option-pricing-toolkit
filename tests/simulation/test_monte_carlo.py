import pytest
import numpy as np

from option_pricing.payoffs.vanilla import EuropeanPut, EuropeanCall, AsianCall
from option_pricing.simulation.results import (
    MonteCarloPricingResult, 
    SimulationResult, 
    AntitheticSimulationResult)
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
    with pytest.raises(TypeError, match="simulation_result must be a SimulationResult."):
        pricer.price("invalid_input")


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
        np.array([0.0, 1.0, 2.0]))

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


def test_price_antithetic_terminal_payoff(pricer):
    positive_paths = np.array([
        [100.0, 120.0],
        [100.0, 110.0],
        [100.0, 130.0],
    ])
    negative_paths = np.array([
        [100.0, 80.0],
        [100.0, 95.0],
        [100.0, 90.0],
    ])
    time_grid = np.array([0.0, 1.0])

    simulation_result = AntitheticSimulationResult(
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        time_grid=time_grid,
    )

    result = pricer.price_antithetic(simulation_result)

    expected_paired_payoffs = np.array([
        10.0,
        5.0,
        15.0,
    ])

    expected_price = np.mean(expected_paired_payoffs)

    expected_standard_error = (
        np.std(expected_paired_payoffs, ddof=1)
        / np.sqrt(expected_paired_payoffs.size)
    )

    assert result.price == pytest.approx(expected_price)
    assert result.standard_error == pytest.approx(expected_standard_error)
    assert result.n_paths == 6


def test_price_antithetic_rejects_invalid_result(pricer):
    with pytest.raises(TypeError):
        pricer.price_antithetic("invalid")


def test_price_antithetic_path_dependent_payoff():
    payoff = AsianCall(strike=100.0)

    pricer = MonteCarloPricer(
        payoff=payoff,
        rate=0.0,
        maturity=1.0,
    )

    positive_paths = np.array([
        [100.0, 110.0, 120.0],
        [100.0, 105.0, 115.0],
    ])

    negative_paths = np.array([
        [100.0, 90.0, 80.0],
        [100.0, 95.0, 105.0],
    ])

    time_grid = np.array([0.0, 0.5, 1.0])

    simulation_result = AntitheticSimulationResult(
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        time_grid=time_grid,
    )

    result = pricer.price_antithetic(simulation_result)

    expected_paired_payoffs = np.array([
        7.5,
        5.0,
    ])

    expected_price = np.mean(expected_paired_payoffs)

    expected_standard_error = (
        np.std(expected_paired_payoffs, ddof=1)
        / np.sqrt(expected_paired_payoffs.size)
    )

    assert result.price == pytest.approx(expected_price)
    assert result.standard_error == pytest.approx(expected_standard_error)
    assert result.n_paths == 4


def test_price_rejects_mismatched_maturity(pricer):
    simulation_result = SimulationResult(
        paths=np.array([
            [100.0, 105.0, 110.0],
            [100.0, 110.0, 120.0],
        ]),
        time_grid=np.array([0.0, 0.5, 2.0]),
    )

    with pytest.raises(ValueError, match="Simulation maturity must match pricer maturity."):
        pricer.price(simulation_result)


def test_price_antithetic_rejects_mismatched_maturity(pricer):
    simulation_result = AntitheticSimulationResult(
        positive_paths=np.array([
            [100.0, 110.0],
            [100.0, 120.0],
        ]),
        negative_paths=np.array([
            [100.0, 90.0],
            [100.0, 80.0],
        ]),
        time_grid=np.array([0.0, 2.0]),
    )

    with pytest.raises(ValueError, match="Simulation maturity must match pricer maturity."):
        pricer.price_antithetic(simulation_result)