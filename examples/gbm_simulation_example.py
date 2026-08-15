import numpy as np

from option_pricing.simulation.gbm import GBM
from option_pricing.simulation.simulator import Simulator



initial_value = 100
volatility = 0.2
rate = 0.05
dividend_yield = 0.02

maturity = 1
n_paths = 100000

# Create the stochastic process
gbm = GBM(initial_value=initial_value, volatility=volatility, rate=rate, dividend_yield=dividend_yield)

# Create the simulator
simulator = Simulator(process=gbm, maturity=maturity, steps=252, n_paths=n_paths, rng=np.random.default_rng(42))

# Run the simulaton
result = simulator.run()

# Extract the terminal values
terminal_values = result.paths[:, -1]

# Monte Carlo sample mean 
sample_mean = np.mean(terminal_values)

# Theoretical Expectation
theoretical_mean = initial_value * np.exp((rate - dividend_yield) * maturity)

# Theoretical variance
theoretical_variance = (initial_value**2 * np.exp(2 * (rate - dividend_yield) * maturity)
                         * (np.exp(volatility**2 * maturity) - 1))

# Monte Carlo standard error
standard_error = np.sqrt(theoretical_variance / n_paths)

# Absolute error
absolute_error = abs(sample_mean - theoretical_mean)

# Standardized error
standardized_error = absolute_error / standard_error

# 95% Confidence Interval
confidence_lower = sample_mean - 1.96 * standard_error
confidence_upper = sample_mean + 1.96 * standard_error

mean_inside_ci = confidence_lower <= theoretical_mean <= confidence_upper


# Print results
print("GBM Monte Carlo Validation")
print("--------------------------")
print(f"Initial value:       {initial_value:.4f}")
print(f"Volatility:          {volatility:.4f}")
print(f"Risk-free rate:      {rate:.4f}")
print(f"Dividend yield:      {dividend_yield:.4f}")
print(f"Maturity:            {maturity:.4f}")
print(f"Paths:               {n_paths:,}")
print()
print(f"Theoretical mean:    {theoretical_mean:.6f}")
print(f"Sample mean:         {sample_mean:.6f}")
print(f"Absolute error:      {absolute_error:.6f}")
print(f"Monte Carlo SE:      {standard_error:.6f}")
print(f"Standardized error:  {standardized_error:.4f}")
print(f"95% CI:              "
      f"[{confidence_lower:.6f}, {confidence_upper:.6f}]")
print(f"Mean inside CI:      {mean_inside_ci}")