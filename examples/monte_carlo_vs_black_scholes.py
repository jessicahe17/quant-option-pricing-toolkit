import numpy as np
from datetime import date

from option_pricing.instruments import (
    Equity,
    ExerciseStyle,
    Option,
    OptionType,
)
from option_pricing.market import MarketData, MarketEnvironment
from option_pricing.models import BlackScholesModel
from option_pricing.payoffs import EuropeanCall
from option_pricing.simulation import GBM, MonteCarloPricer, Simulator


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
n_paths = 100_000
seed = 42



# Black-Scholes price
aapl = Equity("AAPL", "NASDAQ")
option = Option(underlying=aapl, 
                strike=K, 
                expiration_date=expiration_date, 
                option_type=OptionType.CALL, 
                exercise_style=ExerciseStyle.EUROPEAN)

market_data = MarketData(spot=S0, volatility=sigma, risk_free_rate=r, dividend_yield=q)
environment = MarketEnvironment({aapl: market_data}, valuation_date)

bs_model = BlackScholesModel()
bs_result = bs_model.evaluate(option=option, market=environment)
bs_price = bs_result.price



# Monte Carlo simulation
rng = np.random.default_rng(seed)
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


# Monte Carlo pricing
payoff = EuropeanCall(strike=K)
pricer = MonteCarloPricer(payoff=payoff, rate=r, maturity=T)
mc_result = pricer.price(simulation_result)



# Validation
price_difference = mc_result.price - bs_price
absolute_difference = abs(price_difference)
standardized_error = (price_difference / mc_result.standard_error)

z = 1.96
lower_bound = (mc_result.price - z * mc_result.standard_error)
upper_bound = (mc_result.price + z * mc_result.standard_error)

bs_inside_ci = lower_bound <= bs_price <= upper_bound



# Output
print("Monte Carlo vs Black-Scholes Validation")
print("----------------------------------------")
print(f"Black-Scholes price:      {bs_price:.6f}")
print(f"Monte Carlo price:        {mc_result.price:.6f}")
print(f"Standard error:           {mc_result.standard_error:.6f}")
print(f"Absolute difference:      {absolute_difference:.6f}")
print(f"Standardized error:       {standardized_error:.4f}")
print(
    f"95% confidence interval:  "
    f"[{lower_bound:.6f}, {upper_bound:.6f}]"
)
print(f"BS price inside CI:       {bs_inside_ci}")