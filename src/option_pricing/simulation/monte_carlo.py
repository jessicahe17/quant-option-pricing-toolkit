import numpy as np

from option_pricing.payoffs.base import Payoff, PayoffRequirement
from option_pricing.simulation.results import (
    MonteCarloPricingResult,
    SimulationResult,
    AntitheticSimulationResult,
)


class MonteCarloPricer:
    """Price an option using Monte Carlo simulation results."""

    def __init__(self, payoff: Payoff, rate: float, maturity: float) -> None:
        if not isinstance(payoff, Payoff):
            raise TypeError("payoff must be a Payoff.")

        if isinstance(rate, (bool, np.bool_)) or not isinstance(
            rate, (int, float, np.integer, np.floating)
        ):
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
                f"Unsupported payoff requirement: {self.payoff.requirement}"
            )

        payoff_values = self.payoff(values)

        discount_factor = np.exp(-self.rate * self.maturity)

        mean_payoff = np.mean(payoff_values)
        price = discount_factor * mean_payoff

        sample_std = np.std(payoff_values, ddof=1)
        standard_error = (
            discount_factor * sample_std / np.sqrt(simulation_result.paths.shape[0])
        )

        return MonteCarloPricingResult(
            price=float(price),
            standard_error=float(standard_error),
            n_paths=simulation_result.paths.shape[0],
        )


    def price_antithetic(
        self, simulation_result: AntitheticSimulationResult
    ) -> MonteCarloPricingResult:
        """Estimate the option price using antithetic simulation results."""

        if not isinstance(simulation_result, AntitheticSimulationResult):
            raise TypeError("simulation_result must be an AntitheticSimulationResult.")

        if self.payoff.requirement == PayoffRequirement.TERMINAL:
            positive_values = simulation_result.positive_paths[:, -1]
            negative_values = simulation_result.negative_paths[:, -1]

        elif self.payoff.requirement == PayoffRequirement.PATH:
            positive_values = simulation_result.positive_paths
            negative_values = simulation_result.negative_paths

        else:
            raise ValueError(
                f"Unsupported payoff requirement: {self.payoff.requirement}"
            )

        positive_payoffs = self.payoff(positive_values)
        negative_payoffs = self.payoff(negative_values)

        paired_payoffs = 0.5 * (positive_payoffs + negative_payoffs)

        discount_factor = np.exp(-self.rate * self.maturity)

        mean_payoff = np.mean(paired_payoffs)
        price = discount_factor * mean_payoff

        sample_std = np.std(paired_payoffs, ddof=1)
        n_pairs = paired_payoffs.size

        standard_error = (discount_factor * sample_std / np.sqrt(n_pairs))

        return MonteCarloPricingResult(
            price=float(price),
            standard_error=float(standard_error),
            n_paths=2 * n_pairs
        )
