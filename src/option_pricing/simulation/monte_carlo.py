import numpy as np

from option_pricing.payoffs.base import Payoff, PayoffRequirement
from option_pricing.simulation.results import MonteCarloPricingResult, SimulationResult


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



    def price(self, simulation_result: SimulationResult) -> MonteCarloPricingResult:
        """Estimate the option price from simulated paths."""

        if not isinstance(simulation_result, SimulationResult):
            raise TypeError("simulation_result must be a SimulationResult.")

        if self.payoff.requirement == PayoffRequirement.TERMINAL:
            values = simulation_result.paths[:, -1]

        elif self.payoff.requirement == PayoffRequirement.PATH:
            values = simulation_result.paths

        else:
            raise ValueError(
                f"Unsupported payoff requirement: "
                f"{self.payoff.requirement}"
                )

        payoff_values = self.payoff(values)

        discount_factor = np.exp(-self.rate * self.maturity)

        mean_payoff = np.mean(payoff_values)
        price = discount_factor * mean_payoff

        sample_std = np.std(payoff_values, ddof=1)
        standard_error = discount_factor * sample_std / np.sqrt(simulation_result.paths.shape[0])

        return MonteCarloPricingResult(
            price=float(price),
            standard_error=float(standard_error),
            n_paths=simulation_result.paths.shape[0]
            )