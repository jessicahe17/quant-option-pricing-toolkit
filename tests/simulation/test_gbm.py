import pytest
import numpy as np
from option_pricing.simulation.gbm import GBM


def test_valid_creation():
    gbm = GBM(initial_value=100, volatility=0.2, rate=0.05)

    assert gbm.initial_value == 100
    assert gbm.volatility == 0.2
    assert gbm.rate == 0.05
    assert gbm.dividend_yield == 0.0


def test_negative_rate():
    gbm = GBM(initial_value=100.0, volatility=0.2, rate=-0.01)

    assert gbm.rate == -0.01


def test_invalid_initial_value():
    with pytest.raises(ValueError, match="initial_value must be positive"):
        GBM(initial_value=0, volatility=0.2, rate=0.05)

    with pytest.raises(ValueError, match="initial_value must be positive"):
        GBM(initial_value=-1, volatility=0.2, rate=0.05)


def test_negative_volatility():
    with pytest.raises(ValueError, match="volatility must be non-negative"):
        GBM(initial_value=100, volatility=-0.1, rate=0.05)


def test_invalid_dt():
    gbm = GBM(initial_value=100.0, volatility=0.2, rate=0.05)

    with pytest.raises(ValueError, match="Time increment dt must be strictly positive"):
        gbm.step(current_value=100.0, dt=0, random_draw=0.5)

    with pytest.raises(ValueError, match="Time increment dt must be strictly positive"):
        gbm.step(current_value=100.0, dt=-0.5, random_draw=0.5)


def test_zero_volatility():
    S_t = 100.0
    r = 0.05
    q = 0.02
    dt = 0.5
    Z = 0.8  # Should have no effect

    gbm = GBM(initial_value=S_t, volatility=0.0, rate=r, dividend_yield=q)
    next_S = gbm.step(current_value=S_t, dt=dt, random_draw=Z)

    expected_next_S = S_t * np.exp((r - q) * dt)

    assert np.isclose(next_S, expected_next_S)


def test_known_one_step_calculation():
    S_t = 100.0
    r = 0.05
    q = 0.02
    volatility = 0.2
    dt = 0.5
    Z = 0.5

    gbm = GBM(initial_value=S_t, volatility=volatility, rate=r, dividend_yield=q)
    next_S = gbm.step(current_value=S_t, dt=dt, random_draw=Z)

    drift = r - q - 0.5 * volatility**2
    diffusion = volatility * np.sqrt(dt) * Z
    expected_next_S = S_t * np.exp(drift * dt + diffusion)

    assert np.isclose(next_S, expected_next_S)


def test_vectorized_input():
    r = 0.05
    q = 0.02
    volatility = 0.2
    dt = 0.5

    current_values = np.array([100.0, 110.0, 120.0])
    random_draws = np.array([0.5, -0.5, 0.0])

    gbm = GBM(initial_value=100.0, volatility=volatility, rate=r, dividend_yield=q)
    next_values = gbm.step(current_value=current_values, dt=dt, random_draw=random_draws)

    drift = r - q - 0.5 * volatility**2
    diffusion = volatility * np.sqrt(dt) * random_draws
    expected_next_values = current_values * np.exp(drift * dt + diffusion)

    assert isinstance(next_values, np.ndarray)
    assert next_values.shape == current_values.shape
    np.testing.assert_allclose(next_values, expected_next_values)
