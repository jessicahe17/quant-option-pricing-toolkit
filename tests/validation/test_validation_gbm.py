import numpy as np
import pytest
from scipy import stats
from option_pricing.validation.gbm import (
    theoretical_mean, 
    theoretical_variance,
    validate_gbm_moments,
    theoretical_log_mean,
    theoretical_log_std,
    validate_gbm_distribution
    )


def test_theoretical_mean():
    result = theoretical_mean(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.02,
        time_to_maturity=1.0
        )
    expected = 100.0 * np.exp(0.03)

    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_theoretical_variance():
    result = theoretical_variance(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.20,
        time_to_maturity=1.0
        )
    expected = 100.0**2 * np.exp(2 * 0.03) * (np.exp(0.2**2) - 1)

    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_validate_gbm_moments():
    terminal_prices = np.array([98.0, 100.0, 102.0, 104.0, 96.0])

    result = validate_gbm_moments(
        terminal_prices=terminal_prices,
        theoretical_mean=100.0,
        theoretical_variance=10.0
        )

    assert result.sample_mean == 100.0
    assert result.theoretical_mean == 100.0
    assert result.sample_variance == 10.0
    assert result.theoretical_variance == 10.0


def test_validate_gbm_moments_standard_error():
    terminal_prices = np.array([98.0, 100.0, 102.0, 104.0, 96.0])

    result = validate_gbm_moments(
        terminal_prices=terminal_prices,
        theoretical_mean=100.0,
        theoretical_variance=10.0
        )

    expected_standard_error = np.sqrt(10.0 / 5)

    np.testing.assert_allclose(result.standard_error, expected_standard_error)
    np.testing.assert_allclose(result.standardized_error, 0.0)


def test_validate_gbm_moments_rejects_non_1d_input():
    terminal_prices = np.array([[98.0, 100.0], [102.0, 104.0]])

    with pytest.raises(ValueError, match="terminal_prices must be one-dimensional."):
        validate_gbm_moments(
            terminal_prices=terminal_prices, 
            theoretical_mean=100.0,
            theoretical_variance=10.0
            )


def test_validate_gbm_moments_requires_at_least_two_prices():
    terminal_prices = np.array([100.0])

    with pytest.raises(ValueError, match="At least two terminal prices are required."):
        validate_gbm_moments(
            terminal_prices=terminal_prices,
            theoretical_mean=100.0,
            theoretical_variance=10.0
            )


def test_theoretical_log_mean():
    result = theoretical_log_mean(
        spot = 100.0,
        rate = 0.05,
        dividend_yield = 0.02,
        volatility = 0.20,
        time_to_maturity = 1.0
        )
    expected = np.log(100) + 0.01
    np.testing.assert_allclose(result, expected)


def test_theoretical_log_std():
    result = theoretical_log_std(volatility=0.2, time_to_maturity=1)
    expected = 0.2

    np.testing.assert_allclose(result, expected)


def test_validate_gbm_distribution():
    theoretical_log_mean = 4.615
    theoretical_log_std = 0.20
    rng = np.random.default_rng(42)
    log_prices = rng.normal(
        loc=theoretical_log_mean,
        scale=theoretical_log_std,
        size=10_000
        )
    terminal_prices = np.exp(log_prices)
    result = validate_gbm_distribution(
        terminal_prices,
        theoretical_log_mean,
        theoretical_log_std
        )
    normal_cdf = stats.norm(loc=theoretical_log_mean, scale=theoretical_log_std).cdf
    expected_statistic, expected_p_value = stats.kstest(log_prices, normal_cdf)

    np.testing.assert_allclose(result.ks_statistic, expected_statistic, rtol=1e-4)
    np.testing.assert_allclose(result.ks_p_value, expected_p_value, rtol=1e-4)


def test_non_1d_array():
    prices_2d = np.array([[1.0, 1.5], [2.0, 2.5]])
    with pytest.raises(ValueError, match="terminal_prices must be one-dimensional."):
        validate_gbm_distribution(prices_2d, 0.0, 1.0)
        

def test_fewer_than_two_observations():
    prices_one = np.array([100.0])
    with pytest.raises(ValueError, match="At least two terminal prices are required"):
        validate_gbm_distribution(prices_one, 0.0, 1.0)


def test_rejects_zero_prices():
    prices_with_zero = np.array([100.0, 0.0, 105.0])
    with pytest.raises(ValueError, match="terminal_prices must be strictly positive."):
        validate_gbm_distribution(prices_with_zero, 0.0, 1.0)


def test_rejects_negative_prices():
    prices_with_negative = np.array([100.0, -5.0, 105.0])
    with pytest.raises(ValueError, match="terminal_prices must be strictly positive."):
        validate_gbm_distribution(prices_with_negative, 0.0, 1.0)


def test_rejects_non_finite_values():
    prices_with_nan = np.array([100.0, np.nan, 105.0])
    with pytest.raises(ValueError, match="terminal_prices must contain only finite values."):
        validate_gbm_distribution(prices_with_nan, 0.0, 1.0)
    prices_with_inf = np.array([100.0, np.inf, 105.0])
    with pytest.raises(ValueError, match="terminal_prices must contain only finite values."):
        validate_gbm_distribution(prices_with_inf, 0.0, 1.0)


def test_rejects_non_positive_theoretical_log_std():
    valid_prices = np.array([100.0, 105.0, 110.0])
    with pytest.raises(ValueError, match="theoretical_log_std must be strictly positive."):
        validate_gbm_distribution(valid_prices, 0.0, 0.0)
    with pytest.raises(ValueError, match="theoretical_log_std must be strictly positive."):
        validate_gbm_distribution(valid_prices, 0.0, -1.5)