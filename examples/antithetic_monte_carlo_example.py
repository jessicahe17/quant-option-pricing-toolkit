import numpy as np

from option_pricing.payoffs.vanilla import EuropeanCall
from option_pricing.simulation.gbm import GBM
from option_pricing.simulation.monte_carlo import MonteCarloPricer
from option_pricing.simulation.simulator import (
    AntitheticSimulator,
    Simulator,
)



# Parameters
spot = 100.0
strike = 100.0
volatility = 0.20
rate = 0.05
dividend_yield = 0.02
maturity = 1.0

steps = 252
n_paths = 100_000
seed = 42


# Process
process = GBM(
    initial_value=spot,
    volatility=volatility,
    rate=rate,
    dividend_yield=dividend_yield,
)


# Create payoff and pricer
payoff = EuropeanCall(strike=strike)

pricer = MonteCarloPricer(
    payoff=payoff,
    rate=rate,
    maturity=maturity,
)


# Create RNGs
standard_rng = np.random.default_rng(seed)
antithetic_rng = np.random.default_rng(seed)


# Run standard Monte Carlo
standard_simulator = Simulator(
    process=process,
    maturity=maturity,
    steps=steps,
    n_paths=n_paths,
    rng=standard_rng,
)

standard_simulation = standard_simulator.run()

standard_result = pricer.price(standard_simulation)


# Run antithetic Monte Carlo
antithetic_simulator = AntitheticSimulator(
    process=process,
    maturity=maturity,
    steps=steps,
    n_paths=n_paths,
    rng=antithetic_rng,
)

antithetic_simulation = antithetic_simulator.run()

antithetic_result = pricer.price_antithetic(antithetic_simulation)


# Print results
print("Standard Monte Carlo")
print(f"Price:          {standard_result.price:.6f}")
print(f"Standard error: {standard_result.standard_error:.6f}")
print(f"Paths:          {standard_result.n_paths:,}")

print()

print("Antithetic Monte Carlo")
print(f"Price:          {antithetic_result.price:.6f}")
print(f"Standard error: {antithetic_result.standard_error:.6f}")
print(f"Paths:          {antithetic_result.n_paths:,}")