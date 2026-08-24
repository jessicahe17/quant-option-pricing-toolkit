import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from option_pricing.simulation import GBM, Simulator
from option_pricing.validation import (
    theoretical_log_mean,
    theoretical_log_std,
    validate_gbm_distribution,
)



# Parameters
spot = 100.0
volatility = 0.20
rate = 0.05
dividend_yield = 0.02
maturity = 1.0
steps = 252
n_paths = 100_000
seed = 42


# Create the process
process = GBM(
    initial_value=spot,
    volatility=volatility,
    rate=rate,
    dividend_yield=dividend_yield
    )


# Random-number generator
rng = np.random.default_rng(seed)


# Simulator
simulator = Simulator(
    process=process,
    maturity=maturity,
    steps=steps,
    n_paths=n_paths,
    rng=rng
    )


# Run the simulation
simulation_result = simulator.run()


# Extract the terminal values
terminal_prices = simulation_result.paths[:, -1]


# Theoretical log mean and std
log_mean = theoretical_log_mean(
    spot=spot,
    rate=rate,
    dividend_yield=dividend_yield,
    volatility=volatility,
    time_to_maturity=maturity,
    )

log_std = theoretical_log_std(
    volatility=volatility,
    time_to_maturity=maturity
    )



# Compute result
result = validate_gbm_distribution(
    terminal_prices=terminal_prices,
    theoretical_log_mean=log_mean,
    theoretical_log_std=log_std
    )



# Print results
print(f"Theoretical log mean: {result.theoretical_log_mean:.6f}")
print(f"Theoretical log std:  {result.theoretical_log_std:.6f}")
print(f"KS statistic:         {result.ks_statistic:.6f}")
print(f"KS p-value:           {result.ks_p_value:.6f}")



# Plot the empirical log prices against the theoretical PDF
log_prices = np.log(terminal_prices)
x = np.linspace(log_prices.min(), log_prices.max(), 500)
theoretical_pdf = stats.norm.pdf(x, loc=log_mean, scale=log_std)

plt.hist(
    log_prices,
    bins=50,
    density=True,
    alpha=0.6,
    label="Simulated log-prices"
    )

plt.plot(
    x,
    theoretical_pdf,
    linewidth=2,
    label="Theoretical normal PDF"
    )

plt.xlabel("Log terminal price")
plt.ylabel("Density")
plt.title("GBM Terminal Log-Price Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("docs/figures/gbm_terminal_log_prices_distribution.png", dpi=300, bbox_inches="tight")
plt.show()



# Q-Q plot
fig, ax = plt.subplots()

stats.probplot(
    log_prices,
    dist=stats.norm(
        loc=log_mean,
        scale=log_std
        ),
    plot=ax
    )

ax.set_title("Q-Q Plot of GBM Terminal Log-Prices")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")

plt.tight_layout()
plt.savefig("docs/figures/q_q_plot_gbm_terminal_log_prices.png", dpi=300, bbox_inches="tight")
plt.show()
