import numpy as np
import pytest

from option_pricing.simulation.gbm import GBM
from option_pricing.simulation.simulator import Simulator, AntitheticSimulator
from option_pricing.simulation.results import SimulationResult, AntitheticSimulationResult


@pytest.fixture
def gbm():
    return GBM(initial_value=100.0, volatility=0.2, rate=0.05, dividend_yield=0.02)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simulator(gbm, rng):
    return Simulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=rng)


@pytest.fixture
def antithetic_simulator(gbm, rng):
    return AntitheticSimulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=rng)


def test_simulator_valid_creation(gbm, rng, simulator):
    assert simulator.process is gbm
    assert simulator.maturity == 1.0
    assert simulator.steps == 10
    assert simulator.n_paths == 100
    assert simulator.rng is rng


def test_invalid_process(rng):
    with pytest.raises(TypeError, match="process must be a StochasticProcess"):
        Simulator(process="not a process", maturity=1.0, steps=10, n_paths=100, rng=rng)


def test_invalid_maturity(gbm, rng):
    with pytest.raises(ValueError, match="maturity must be strictly positive"):
        Simulator(process=gbm, maturity=0.0, steps=10, n_paths=100, rng=rng)

    with pytest.raises(ValueError, match="maturity must be strictly positive"):
        Simulator(process=gbm, maturity=-1.0, steps=10, n_paths=100, rng=rng)


def test_invalid_steps(gbm, rng):
    with pytest.raises(TypeError, match="steps must be an integer"):
        Simulator(process=gbm, maturity=1.0, steps=10.5, n_paths=100, rng=rng)

    with pytest.raises(ValueError, match="steps must be at least 1"):
        Simulator(process=gbm, maturity=1.0, steps=0, n_paths=100, rng=rng)


def test_invalid_n_paths(gbm, rng):
    with pytest.raises(TypeError, match="n_paths must be an integer"):
        Simulator(process=gbm, maturity=1.0, steps=5, n_paths=100.5, rng=rng)

    with pytest.raises(ValueError, match="n_paths must be at least 1"):
        Simulator(process=gbm, maturity=1.0, steps=5, n_paths=0, rng=rng)


def test_invalid_rng(gbm):
    with pytest.raises(TypeError, match="rng must be a NumPy Generator"):
        Simulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=np.random)


def test_run_returns_simulation_result(simulator):
    result = simulator.run()
    assert isinstance(result, SimulationResult)


def test_paths_shape(simulator):
    result = simulator.run()
    assert result.paths.shape == (100, 11)


def test_time_grid_shape(simulator):
    result = simulator.run()
    assert result.time_grid.shape == (11,)


def test_initial_values(gbm, simulator):
    result = simulator.run()
    np.testing.assert_allclose(result.paths[:, 0], gbm.initial_value)


def test_time_grid(gbm, rng):
    maturity = 1.0
    steps = 4
    simulator = Simulator(process=gbm, maturity=maturity, steps=steps, n_paths=10, rng=rng)
    result = simulator.run()

    expected_time_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    np.testing.assert_allclose(result.time_grid, expected_time_grid)


def test_time_grid_ends_at_maturity(gbm, rng):
    maturity = 2.0
    simulator = Simulator(process=gbm, maturity=maturity, steps=8, n_paths=10, rng=rng)
    result = simulator.run()

    assert result.time_grid[0] == 0.0
    assert result.time_grid[-1] == maturity


def test_simulator_reproducibility(gbm):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    simulator1 = Simulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=rng1)
    simulator2 = Simulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=rng2)

    result1 = simulator1.run()
    result2 = simulator2.run()

    np.testing.assert_array_equal(result1.paths, result2.paths)
    np.testing.assert_array_equal(result1.time_grid, result2.time_grid)


def test_different_seeds_produce_different_paths(gbm):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(123)

    simulator1 = Simulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=rng1)
    simulator2 = Simulator(process=gbm, maturity=1.0, steps=10, n_paths=100, rng=rng2)

    result1 = simulator1.run()
    result2 = simulator2.run()

    assert not np.array_equal(result1.paths, result2.paths)


def test_gbm_paths_are_positive(simulator):
    result = simulator.run()
    assert np.all(result.paths > 0)


def test_antithetic_simulator_valid_creation(gbm, rng, antithetic_simulator):
    assert antithetic_simulator.process is gbm
    assert antithetic_simulator.maturity == 1.0
    assert antithetic_simulator.steps == 10
    assert antithetic_simulator.n_paths == 100
    assert antithetic_simulator.rng is rng


def test_reject_odd_n_paths(gbm, rng):
    with pytest.raises(ValueError):
        AntitheticSimulator(process=gbm, maturity=1.0, steps=10, n_paths=5, rng=rng)


def test_output_shapes(antithetic_simulator):
    result = antithetic_simulator.run()
    n_pairs = antithetic_simulator.n_paths // 2
    n_steps = antithetic_simulator.steps + 1
    expected_shape = (n_pairs, n_steps)

    assert result.positive_paths.shape == expected_shape
    assert result.negative_paths.shape == expected_shape
    assert result.time_grid.shape == (n_steps,)


def test_antithetic_simulator_reproducibility(gbm):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    simulator1 = AntitheticSimulator(process=gbm, maturity=1, steps=10, n_paths=4, rng=rng1)
    simulator2 = AntitheticSimulator(process=gbm, maturity=1, steps=10, n_paths=4, rng=rng2)

    result1 = simulator1.run()
    result2 = simulator2.run()

    np.testing.assert_array_equal(result1.positive_paths, result2.positive_paths)
    np.testing.assert_array_equal(result1.negative_paths, result2.negative_paths)
    np.testing.assert_array_equal(result1.time_grid, result2.time_grid)


def test_antithetic_simulator_pairs_opposite_random_draws(gbm, rng):
    maturity = 1.0
    simulator = AntitheticSimulator(
        process=gbm,
        maturity=maturity,
        steps=1,
        n_paths=4,
        rng=rng
        )

    result = simulator.run()

    positive_terminal = result.positive_paths[:, -1]
    negative_terminal = result.negative_paths[:, -1]

    drift = (gbm.rate - gbm.dividend_yield - 0.5 * gbm.volatility**2)

    expected_product = (gbm.initial_value**2 * np.exp(2 * drift * maturity))

    np.testing.assert_allclose(
        positive_terminal * negative_terminal,
        expected_product
        )