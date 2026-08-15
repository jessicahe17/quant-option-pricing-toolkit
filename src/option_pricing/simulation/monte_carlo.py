import numpy as np

from option_pricing.payoffs.base import Payoff
from option_pricing.simulation.results import MonteCarloPricingResult


class MonteCarloPricer:
    """Price an option using Monte Carlo simulation results."""

    def __init__(self, payoff: Payoff, rate: float, maturity: float) -> None:
        if not isinstance(payoff, Payoff):
            raise TypeError("payoff must be a Payoff.")

        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, float, np.integer, np.floating)):
            raise TypeError("rate must be a numeric value.")
        if not np.isfinite(rate):
            raise ValueError("rate must be a finite number.")

        if not isinstance(maturity, (int, float, np.integer, np.floating)):
            raise TypeError("maturity must be a numeric value.")
        if not np.isfinite(maturity):
            raise ValueError("maturity must be a finite number.")
        if maturity <= 0:
            raise ValueError("maturity must be strictly positive.")

        self.payoff = payoff
        self.rate = rate
        self.maturity = maturity



    def price(self, terminal_values: np.ndarray) -> MonteCarloPricingResult:
        """Estimate the option price from simulated terminal values."""

        if not isinstance(terminal_values, np.ndarray):
            raise TypeError("terminal_values must be a NumPy array.")

        if terminal_values.ndim != 1:
            raise ValueError("terminal_values must be a 1-D array.")

        if terminal_values.size < 2:
            raise ValueError("terminal_values must contain at least two values.")

        if not np.issubdtype(terminal_values.dtype, np.number):
            raise TypeError("terminal_values must contain numeric values.")

        if not np.all(np.isfinite(terminal_values)):
            raise ValueError("terminal_values must contain only finite values.")


        payoff_values = self.payoff(terminal_values)

        discount_factor = np.exp(-self.rate * self.maturity)

        mean_payoff = np.mean(payoff_values)
        price = discount_factor * mean_payoff

        sample_std = np.std(payoff_values, ddof=1)
        standard_error = discount_factor * sample_std / np.sqrt(terminal_values.size)

        return MonteCarloPricingResult(
            price=float(price),
            standard_error=float(standard_error),
            n_paths=terminal_values.size
            )