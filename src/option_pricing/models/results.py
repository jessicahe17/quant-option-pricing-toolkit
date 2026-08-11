from dataclasses import dataclass


@dataclass(frozen=True)
class Greeks:
    """
    Immutable container for option pricing sensitivities.

    Stores the risk metrics associated with a financial derivative.
    Contains no calculation logic.

    Parameters
    ----------
    delta : float
        Sensitivity of option value to changes in the underlying price.

    gamma : float
        Sensitivity of delta to changes in the underlying price.

    vega : float
        Sensitivity of option value to changes in volatility.

    theta : float
        Sensitivity of option value to the passage of time.

    rho : float
        Sensitivity of option value to changes in the risk-free rate.
    """

    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None



@dataclass(frozen=True)
class PricingResult:
    """
    Immutable output from an option pricing model.

    Contains the theoretical option value and associated risk sensitivities.

    Parameters
    ----------
    price : float
        Theoretical fair value of the option.

    greeks : Greeks
        Collection of option price sensitivities.
    """

    price: float
    greeks: Greeks