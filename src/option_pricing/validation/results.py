from dataclasses import dataclass


@dataclass(frozen=True)
class GBMMomentValidationResult:
    """Summary of empirical and theoretical GBM moment validation."""

    sample_mean: float
    theoretical_mean: float
    sample_variance: float
    theoretical_variance: float
    standard_error: float
    standardized_error: float



@dataclass(frozen=True)
class GBMDistributionValidationResult:
    """Summary of GBM terminal-log-price distribution validation."""

    theoretical_log_mean: float
    theoretical_log_std: float
    ks_statistic: float
    ks_p_value: float