import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from option_pricing.instruments.underlying import Equity
from option_pricing.instruments.option import Option, OptionType, ExerciseStyle
from option_pricing.market.market_data import MarketData
from option_pricing.market.market_environment import MarketEnvironment
from option_pricing.models.black_scholes import BlackScholesModel
from option_pricing.payoffs.vanilla import EuropeanCall
from option_pricing.simulation.gbm import GBM
from option_pricing.simulation.simulator import Simulator
from option_pricing.simulation.monte_carlo import MonteCarloPricer


# Parameters
S0 = 100.0
K = 100.0
sigma = 0.20
r = 0.05
q = 0.02

expiration_date = date(2027, 1, 1)
valuation_date = date(2026, 1, 1)
T = (expiration_date - valuation_date).days / 365

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



# Black-Scholes benchmark
aapl = Equity("AAPL", "NASDAQ")
option = Option(underlying=aapl,
                strike=K,
                expiration_date=expiration_date,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
                )

market_data = MarketData(spot=S0, volatility=sigma, risk_free_rate=r, dividend_yield=q)
environment = MarketEnvironment({aapl: market_data}, valuation_date)

bs_model = BlackScholesModel()
bs_result = bs_model.evaluate(option=option, market=environment)
bs_price = bs_result.price



# Monte Carlo convergence experiment
payoff = EuropeanCall(strike=K)
pricer = MonteCarloPricer(payoff=payoff, rate=r, maturity=T)
rng = np.random.default_rng(seed)

results = []

for n_paths in path_counts:

    process = GBM(
        initial_value=S0,
        volatility=sigma,
        rate=r,
        dividend_yield=q
        )

    simulator = Simulator(
        process=process,
        maturity=T,
        steps=steps,
        n_paths=n_paths,
        rng=rng
        )

    simulation_result = simulator.run()

    terminal_values = simulation_result.paths[:, -1]

    mc_result = pricer.price(terminal_values)

    absolute_error = abs(mc_result.price - bs_price)

    standardized_error = ((mc_result.price - bs_price) / mc_result.standard_error)

    scaled_standard_error = (mc_result.standard_error * np.sqrt(n_paths))

    results.append(
        (
            n_paths,
            mc_result.price,
            absolute_error,
            mc_result.standard_error,
            standardized_error,
            scaled_standard_error,
            )
        )



# Output
print("Monte Carlo Convergence Study")
print("=============================")
print(f"Black-Scholes price: {bs_price:.6f}")
print()

print(
    f"{'Paths':>10}"
    f"{'MC Price':>15}"
    f"{'Abs. Error':>15}"
    f"{'Std. Error':>15}"
    f"{'Std. Error × √N':>20}"
    )

print("-" * 75)

for (
    n_paths,
    mc_price,
    absolute_error,
    standard_error,
    standardized_error,
    scaled_standard_error,
    ) in results:

    print(
        f"{n_paths:>10,d}"
        f"{mc_price:>15.6f}"
        f"{absolute_error:>15.6f}"
        f"{standard_error:>15.6f}"
        f"{scaled_standard_error:>20.6f}"
        )




path_counts_array = np.array([result[0] for result in results])
mc_prices = np.array([result[1] for result in results])
absolute_errors = np.array([result[2] for result in results])
standard_errors = np.array([result[3] for result in results])



# Calculate the empirical convergence slope.
slope, intercept = np.polyfit(np.log(path_counts_array), np.log(standard_errors), 1)

print()
print(f"Empirical convergence slope: {slope:.4f}")



# Figure 1: mc_prices vs bs_price
plt.figure()

plt.plot(
    path_counts_array,
    mc_prices,
    marker="o",
    label="Monte Carlo"
    )

plt.axhline(
    bs_price,
    linestyle="--",
    label="Black-Scholes"
    )

plt.xscale("log")

plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.title("Monte Carlo Price Convergence")
plt.legend()
plt.grid(True)

plt.savefig("docs/figures/mc_price_convergence.png", dpi=300, bbox_inches="tight")

plt.show()



#Figure 2: SE vs N + N^-1/2 reference
plt.figure()

plt.loglog(
    path_counts_array,
    standard_errors,
    marker="o",
    label="Monte Carlo Standard Error"
    )

reference = (standard_errors[0] * np.sqrt(path_counts_array[0] / path_counts_array))

plt.loglog(
    path_counts_array,
    reference,
    linestyle="--",
    label=r"$N^{-1/2}$ reference"
    )

plt.xlabel("Number of Paths")
plt.ylabel("Standard Error")
plt.title("Monte Carlo Standard Error Convergence")
plt.legend()
plt.grid(True)

plt.savefig("docs/figures/mc_standard_error_convergence.png", dpi=300, bbox_inches="tight")

plt.show()