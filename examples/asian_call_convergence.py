import numpy as np
import matplotlib.pyplot as plt

from option_pricing.payoffs.vanilla import AsianCall
from option_pricing.simulation.monte_carlo import MonteCarloPricer
from option_pricing.simulation.gbm import GBM
from option_pricing.simulation.simulator import Simulator


# parameters
S0 = 100.0
K = 100.0
sigma = 0.20
r = 0.05
q = 0.02
T = 1.0

steps = 252
seed = 42

path_counts = [
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000
    ]


# Create process
process = GBM(
    initial_value=S0,
    volatility=sigma,
    rate=r,
    dividend_yield=q
    )


# Create payoff and pricer
payoff = AsianCall(strike=K)

pricer = MonteCarloPricer(
        payoff=payoff,
        rate=r,
        maturity=T
        )


# Run convergence experiments
results = []

for n_paths in path_counts:
    rng = np.random.default_rng(seed)

    simulator = Simulator(
        process=process,
        maturity=T,
        steps=steps,
        n_paths=n_paths,
        rng=rng
        )


    simulation_result = simulator.run()

    result = pricer.price(simulation_result)

    se_sqrt_n = result.standard_error * np.sqrt(n_paths)

    results.append(
        (
            n_paths,
            result.price,
            result.standard_error,
            se_sqrt_n
            )
        )


# Calculate 95% confidence interval
confidence_level = 1.96

lower_bound = result.price - confidence_level * result.standard_error
upper_bound = result.price + confidence_level * result.standard_error



# Compute the empirical convergence slope
path_counts_array = np.array([row[0] for row in results])
standard_errors = np.array([row[2] for row in results])

log_paths = np.log(path_counts_array)
log_standard_errors = np.log(standard_errors)

slope, intercept = np.polyfit(log_paths, log_standard_errors, 1)



# Display results
LABEL_WIDTH = 27

print("Asian Call Monte Carlo Validation")
print("=" * 35)

print("\nParameters:")
print(f"{'S0:':<{LABEL_WIDTH}}{S0:.2f}")
print(f"{'K:':<{LABEL_WIDTH}}{K:.2f}")
print(f"{'sigma:':<{LABEL_WIDTH}}{sigma:.2f}")
print(f"{'r:':<{LABEL_WIDTH}}{r:.2f}")
print(f"{'q:':<{LABEL_WIDTH}}{q:.2f}")
print(f"{'T:':<{LABEL_WIDTH}}{T:.2f}")
print(f"{'Steps:':<{LABEL_WIDTH}}{steps}")

print("\nConvergence Results:")
print(
    f"{'Paths':>12}"
    f"{'MC Price':>15}"
    f"{'Std. Error':>15}"
    f"{'SE * sqrt(N)':>15}"
)

print("-" * 57)

for n_paths, price, standard_error, se_sqrt_n in results:
    print(
        f"{n_paths:>12,}"
        f"{price:>15.6f}"
        f"{standard_error:>15.6f}"
        f"{se_sqrt_n:>15.6f}"
    )

print("\nConvergence Analysis:")
print(f"{'Empirical slope:':<{LABEL_WIDTH}}{slope:.4f}")
print(f"{'Theoretical slope:':<{LABEL_WIDTH}}{-0.5:.4f}")



# Create log-log convergence plot
plt.figure(figsize=(8, 5))

plt.loglog(
    path_counts_array,
    standard_errors,
    marker="o",
    label="Monte Carlo standard error"
    )

# Fitted convergence line
fitted_errors = np.exp(intercept) * path_counts_array**slope

plt.loglog(
    path_counts_array,
    fitted_errors,
    linestyle="--",
    label=f"Fitted slope = {slope:.4f}"
    )

plt.xlabel("Number of paths")
plt.ylabel("Standard error")
plt.title("Asian Call Monte Carlo Convergence")
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("docs/figures/asian_call_convergence.png", dpi=300, bbox_inches="tight")

plt.show()