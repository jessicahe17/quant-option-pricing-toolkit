import numpy as np
from scipy import stats

from option_pricing.validation.results import GBMMomentValidationResult, GBMDistributionValidationResult



def theoretical_mean(
    spot: float,
    rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    ) -> float:
    """Return the theoretical mean of S_T under risk-neutral GBM."""

    return spot * np.exp((rate - dividend_yield) * time_to_maturity)



def theoretical_variance(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    ) -> float:
    """Return the theoretical variance of S_T under risk-neutral GBM."""

    return (spot**2 * np.exp(2 * (rate - dividend_yield) * time_to_maturity)
            * (np.exp(volatility**2 * time_to_maturity) - 1))



def validate_gbm_moments(
    terminal_prices: np.ndarray,
    theoretical_mean: float,
    theoretical_variance: float,
    ) -> GBMMomentValidationResult:
    """Validate simulated terminal prices against theoretical GBM moments."""

    if terminal_prices.ndim != 1:
        raise ValueError("terminal_prices must be one-dimensional.")

    if terminal_prices.size < 2:
        raise ValueError("At least two terminal prices are required.")

    sample_mean = float(np.mean(terminal_prices))
    sample_variance = float(np.var(terminal_prices, ddof=1))
    standard_error = np.sqrt(sample_variance / terminal_prices.size)

    if standard_error > 0:
        standardized_error = (sample_mean - theoretical_mean) / standard_error
    else:
        standardized_error = 0.0

    return GBMMomentValidationResult(
        sample_mean=sample_mean,
        theoretical_mean=theoretical_mean,
        sample_variance=sample_variance,
        theoretical_variance=theoretical_variance,
        standard_error=standard_error,
        standardized_error=standardized_error,
        )



def theoretical_log_mean(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float
    ) -> float:
    """Return the theoretical mean of ln(S_T) under risk-neutral GBM."""

    return np.log(spot) + (rate - dividend_yield - 0.5 * volatility**2) * time_to_maturity



def theoretical_log_std(
    volatility: float,
    time_to_maturity: float
    ) -> float:
    """Return the theoretical standard deviation of ln(S_T) under GBM."""

    return volatility * np.sqrt(time_to_maturity)



def validate_gbm_distribution(
    terminal_prices: np.ndarray,
    theoretical_log_mean: float,
    theoretical_log_std: float
    ) -> GBMDistributionValidationResult:
    """Validate the terminal log-price distribution against GBM theory."""

    if terminal_prices.ndim != 1:
        raise ValueError("terminal_prices must be one-dimensional.")

    if terminal_prices.size < 2:
        raise ValueError("At least two terminal prices are required.")

    if not np.all(np.isfinite(terminal_prices)):
        raise ValueError("terminal_prices must contain only finite values.")

    if np.any(terminal_prices <= 0):
        raise ValueError("terminal_prices must be strictly positive.")

    if theoretical_log_std <= 0:
        raise ValueError("theoretical_log_std must be strictly positive.")

    log_prices = np.log(terminal_prices)

    normal_cdf = stats.norm(loc=theoretical_log_mean, scale=theoretical_log_std).cdf

    ks_statistic, ks_p_value = stats.kstest(log_prices, normal_cdf)

    return GBMDistributionValidationResult(
        theoretical_log_mean=theoretical_log_mean,
        theoretical_log_std=theoretical_log_std,
        ks_statistic=ks_statistic,
        ks_p_value=ks_p_value
        )