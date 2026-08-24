import numpy as np
import matplotlib.pyplot as plt

from option_pricing.payoffs import EuropeanCall
from option_pricing.simulation import (
    AntitheticSimulator,
    GBM,
    MonteCarloPricer,
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
seed = 42

path_counts = [
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
]


# Create process, payoff and pricer
process = GBM(
    initial_value=spot,
    volatility=volatility,
    rate=rate,
    dividend_yield=dividend_yield,
)

payoff = EuropeanCall(strike=strike)

pricer = MonteCarloPricer(
    payoff=payoff,
    rate=rate,
    maturity=maturity,
)


# Standard vs antithetic experiment
results = []

for n_paths in path_counts:
    standard_rng = np.random.default_rng(seed)
    antithetic_rng = np.random.default_rng(seed)

    standard_simulator = Simulator(
        process=process,
        maturity=maturity,
        steps=steps,
        n_paths=n_paths,
        rng=standard_rng,
    )
    standard_simulation = standard_simulator.run()
    standard_result = pricer.price(standard_simulation)

    antithetic_simulator = AntitheticSimulator(
        process=process,
        maturity=maturity,
        steps=steps,
        n_paths=n_paths,
        rng=antithetic_rng,
    )
    antithetic_simulation = antithetic_simulator.run()
    antithetic_result = pricer.price_antithetic(antithetic_simulation)

    standard_scaled_se = (standard_result.standard_error * np.sqrt(n_paths))

    antithetic_scaled_se = (antithetic_result.standard_error * np.sqrt(n_paths))

    efficiency_gain = (
        standard_result.standard_error**2
        / antithetic_result.standard_error**2
    )

    results.append(
        (
            n_paths,
            standard_result.price,
            standard_result.standard_error,
            standard_scaled_se,
            antithetic_result.price,
            antithetic_result.standard_error,
            antithetic_scaled_se,
            efficiency_gain,
        )
    )


# Compute the convergence slopes
path_array = np.array([result[0] for result in results], dtype=float)
standard_se_array = np.array([result[2] for result in results])
antithetic_se_array = np.array([result[5] for result in results])

standard_slope, _ = np.polyfit(
    np.log(path_array),
    np.log(standard_se_array),
    1,
)
antithetic_slope, _ = np.polyfit(
    np.log(path_array),
    np.log(antithetic_se_array),
    1,
)


# Print results
print(
    f"{'Paths':>10} "
    f"{'Std Price':>12} "
    f"{'Std SE':>10} "
    f"{'Std SE√N':>10} "
    f"{'AV Price':>12} "
    f"{'AV SE':>10} "
    f"{'AV SE√N':>10} "
    f"{'Gain':>8}"
)

for (
    n_paths,
    standard_price,
    standard_se,
    standard_scaled_se,
    antithetic_price,
    antithetic_se,
    antithetic_scaled_se,
    efficiency_gain,
) in results:
    print(
        f"{n_paths:>10,} "
        f"{standard_price:>12.6f} "
        f"{standard_se:>10.6f} "
        f"{standard_scaled_se:>10.6f} "
        f"{antithetic_price:>12.6f} "
        f"{antithetic_se:>10.6f} "
        f"{antithetic_scaled_se:>10.6f} "
        f"{efficiency_gain:>8.4f}"
    )

print()
print(f"Standard MC convergence slope:   {standard_slope:.4f}")
print(f"Antithetic MC convergence slope: {antithetic_slope:.4f}")
print("Theoretical slope:               -0.5000")



# Figure 1: convergence figure
plt.figure(figsize=(8, 5))

plt.loglog(
    path_array,
    standard_se_array,
    marker="o",
    label="Standard Monte Carlo",
)

plt.loglog(
    path_array,
    antithetic_se_array,
    marker="o",
    label="Antithetic Monte Carlo",
)

plt.xlabel("Number of Paths")
plt.ylabel("Standard Error")
plt.title("Standard vs. Antithetic Monte Carlo Convergence")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "docs/figures/antithetic_standard_error_convergence.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()



# Figure 2: efficiency gain
efficiency_array = np.array([result[7] for result in results])

plt.figure(figsize=(8, 5))

plt.plot(
    path_array,
    efficiency_array,
    marker="o",
)

plt.axhline(
    y=1.0,
    linestyle="--",
    label="No efficiency gain",
)

plt.xscale("log")

plt.xlabel("Number of Paths")
plt.ylabel("Variance Efficiency Gain")
plt.title("Antithetic Variance-Reduction Efficiency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "docs/figures/antithetic_efficiency_gain.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()