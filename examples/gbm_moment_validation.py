import numpy as np

from option_pricing.simulation.gbm import GBM
from option_pricing.simulation.simulator import Simulator
from option_pricing.validation.gbm import theoretical_mean, theoretical_variance, validate_gbm_moments



# Parameters
S0 = 100.0
sigma = 0.20
r = 0.05
q = 0.02
T = 1.0

steps = 252
n_paths = 100_000
seed = 42


# Create the process
gbm = GBM(initial_value=S0, volatility=sigma, rate=r, dividend_yield=q)


# Random-number generator
rng = np.random.default_rng(seed)


# Simulator
simulator = Simulator(
    process=gbm,
    maturity=T,
    steps=steps,
    n_paths=n_paths,
    rng=rng
    )


# Run the simulaton
simulation_result = simulator.run()


# Extract the terminal values
terminal_prices = simulation_result.paths[:, -1]


# Calculate the theoretical mean and variance
expected_mean = theoretical_mean(spot=S0, rate=r, dividend_yield=q, time_to_maturity=T)
expected_variance = theoretical_variance(
    spot=S0, 
    rate=r, 
    dividend_yield=q, 
    volatility=sigma, 
    time_to_maturity=T
    )


# Compute result
result = validate_gbm_moments(
    terminal_prices=terminal_prices,
    theoretical_mean=expected_mean,
    theoretical_variance=expected_variance
    )


# Print results
print(f"Sample mean:        {result.sample_mean:.6f}")
print(f"Theoretical mean:   {result.theoretical_mean:.6f}")
print(f"Sample variance:    {result.sample_variance:.6f}")
print(f"Theoretical variance:{result.theoretical_variance:.6f}")
print(f"Standard error:     {result.standard_error:.6f}")
print(f"Standardized error: {result.standardized_error:.6f}")