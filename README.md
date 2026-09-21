# Quantitative Option Pricing Toolkit

A modular Python library for analytical and Monte Carlo option pricing, developed to explore stochastic simulation, numerical validation, path-dependent derivatives, and variance-reduction techniques in quantitative finance.

The project implements Black-Scholes pricing and analytical Greeks, a reusable Geometric Brownian Motion simulation engine, generic Monte Carlo pricing for terminal and path-dependent payoffs, statistical validation of simulated GBM dynamics, and antithetic variates for variance reduction.

## Highlights

- Black-Scholes pricing for European calls and puts with continuous dividend yield
- Analytical Delta, Gamma, Vega, Theta, and Rho
- Exact-discretization risk-neutral Geometric Brownian Motion simulation
- Generic Monte Carlo pricing architecture
- European call and put payoffs
- Arithmetic-average Asian call pricing using full simulated paths
- Monte Carlo convergence analysis against the theoretical $N^{-1/2}$ rate
- Statistical validation of GBM moments and terminal log-price distribution
- Antithetic variates with statistically correct pair-based standard-error estimation
- Reproducible numerical experiments, tests, documentation, and research figures

## Quantitative Results

Key numerical findings from the validation and efficiency experiments:

| Experiment | Result |
| :--- | :--- |
| European MC vs. Black-Scholes | BS price 9.227006 vs. MC price 9.258790 |
| European MC standardized error $(Z)$ | 0.7209 |
| European MC convergence slope | -0.5040 |
| Asian call convergence slope | -0.5039 |
| GBM mean standardized error | 0.6405 |
| GBM terminal log-price KS statistic | 0.002252 |
| Antithetic MC convergence slope | -0.5049 |
| Antithetic variance-efficiency gain | approximately 1.8× |

The experiments support the expected Monte Carlo convergence behavior

$$
\mathrm{SE}_N \propto N^{-1/2},
$$

while showing that antithetic variates reduce the variance constant without changing the asymptotic convergence order.

## Architecture

The library separates financial objects, market state, analytical pricing, stochastic simulation, payoff definitions, Monte Carlo estimation, and statistical validation.

```text
              Financial Instruments
                       │
                       │
                  Market Data
                       │
             ┌─────────┴─────────┐
             │                   │
             ↓                   ↓
      Black-Scholes             GBM
             │                   │
             ↓                   ↓
      PricingResult          Simulator
                                 │
                                 ↓
                         SimulationResult
                                 │
                                 │
Payoff Definitions ──────────────┤
                                 ↓
                         MonteCarloPricer
                                 │
                                 ↓
                    MonteCarloPricingResult

Simulation outputs also feed a separate validation layer for GBM moment, distributional, and numerical-efficiency analysis.
```

Key design principles include:

- Financial contracts and market data are immutable data objects.
- Analytical pricing, simulation, payoff evaluation, and validation are separated by responsibility.
- The Monte Carlo pricer remains generic: payoffs declare whether they require terminal values or full simulated paths.
- Standard and antithetic simulations remain distinct so that antithetic pairing is preserved explicitly.
- Statistical validation is kept separate from the simulation engine.

## Features

### Analytical Pricing

- European call and put pricing under Black-Scholes
- Analytical Delta, Gamma, Vega, Theta, and Rho
- Explicit handling of expiry and zero-volatility cases

### Monte Carlo Simulation

- Exact-transition risk-neutral Geometric Brownian Motion
- Reproducible simulations using NumPy random generators
- Generic terminal-value and path-dependent payoff support
- Monte Carlo price estimates with sample standard errors
- Arithmetic-average Asian call pricing

### Statistical Validation

- Comparison of Monte Carlo European-call prices with Black-Scholes
- Empirical convergence analysis across multiple path budgets
- GBM terminal-price mean and variance validation
- Kolmogorov-Smirnov validation of terminal log-price distributions
- Histogram and Q-Q visual diagnostics

### Variance Reduction

- Explicit antithetic path pairing using $Z$ and $-Z$
- Standard errors computed from independent antithetic pair averages
- Same-path-budget comparison with standard Monte Carlo
- Quantitative variance-efficiency and convergence analysis

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/jessicahe17/quant-option-pricing-toolkit.git
cd quant-option-pricing-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The project requires Python 3.10 or later. Core dependencies are NumPy and SciPy; the development extras include pytest and Matplotlib for testing and numerical experiments.

## Quick Start

The example below prices a European call option with Black-Scholes and reports its analytical Greeks.

```python
from datetime import date

from option_pricing.instruments import (
    Equity,
    ExerciseStyle,
    Option,
    OptionType,
)
from option_pricing.market import MarketData, MarketEnvironment
from option_pricing.models import BlackScholesModel


underlying = Equity(
    symbol="AAPL",
    exchange="NASDAQ",
)

option = Option(
    underlying=underlying,
    strike=100.0,
    expiration_date=date(2027, 1, 1),
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN,
)

market = MarketEnvironment(
    data={
        underlying: MarketData(
            spot=100.0,
            volatility=0.20,
            risk_free_rate=0.05,
            dividend_yield=0.02,
        )
    },
    valuation_date=date(2026, 1, 1),
)

result = BlackScholesModel().evaluate(option, market)

print(f"Price: {result.price:.4f}")
print(f"Delta: {result.greeks.delta:.4f}")
print(f"Gamma: {result.greeks.gamma:.6f}")
print(f"Vega:  {result.greeks.vega:.4f}")
print(f"Theta: {result.greeks.theta:.4f}")
print(f"Rho:   {result.greeks.rho:.4f}")
```

For Monte Carlo pricing, path-dependent examples, GBM validation, and antithetic variance reduction, see the scripts in [`examples/`](examples/).

## Numerical Experiments

The `examples/` directory contains reproducible scripts covering the main numerical experiments in the project:

| Script | Purpose |
| :--- | :--- |
| `black_scholes_example.py` | Black-Scholes pricing and analytical Greeks |
| `monte_carlo_vs_black_scholes.py` | Validate Monte Carlo pricing against the Black-Scholes benchmark |
| `monte_carlo_convergence.py` | Study Monte Carlo price and standard-error convergence |
| `asian_call_convergence.py` | Demonstrate path-dependent Asian option pricing and convergence |
| `gbm_moment_validation.py` | Compare simulated GBM terminal moments with theory |
| `gbm_distribution_validation.py` | Validate the terminal log-price distribution using a KS test, histogram, and Q-Q plot |
| `antithetic_monte_carlo_example.py` | Compare standard and antithetic Monte Carlo for a fixed path budget |
| `antithetic_efficiency_experiment.py` | Quantify variance reduction and convergence under antithetic sampling |

Selected research figures are stored in [`docs/figures/`](docs/figures/), with detailed interpretation in the corresponding documentation.

## Project Structure

```text
quant-option-pricing-toolkit/
├── src/option_pricing/
│   ├── instruments/      # Option contracts and underlyings
│   ├── market/           # Market data and market environments
│   ├── models/           # Pricing models and analytical results
│   ├── payoffs/          # Terminal and path-dependent payoff definitions
│   ├── simulation/       # GBM, simulation engines, and Monte Carlo pricing
│   └── validation/       # Statistical GBM validation utilities
├── tests/                # Unit and regression tests
├── examples/             # Reproducible pricing and research experiments
├── docs/                 # Research notes and numerical validation
│   └── figures/          # Generated research figures
├── pyproject.toml
└── README.md
```

## Testing

The project includes a pytest-based test suite covering:

- financial-object and market-data validation
- Black-Scholes prices and analytical Greeks
- pricing edge cases such as expiry and zero volatility
- GBM simulation and reproducibility
- standard and antithetic simulation behavior
- terminal and path-dependent payoff evaluation
- Monte Carlo pricing and standard-error estimation
- GBM moment and distributional validation
- public package imports

Run the full test suite with:

```bash
python -m pytest
```

## Documentation

Detailed numerical and methodological notes are available in [`docs/`](docs/):

- [`greeks.md`](docs/greeks.md) — analytical Greeks and conventions
- [`monte_carlo_validation.md`](docs/monte_carlo_validation.md) — Monte Carlo validation, convergence, and Asian option pricing
- [`gbm_validation.md`](docs/gbm_validation.md) — moment and distributional validation of simulated GBM
- [`antithetic_variates.md`](docs/antithetic_variates.md) — antithetic variance reduction and efficiency analysis

These documents complement the source code by recording the mathematical assumptions, validation methodology, numerical results, and interpretation of the experiments.

## Scope

This project is designed as a focused quantitative-finance research and software-engineering toolkit rather than a production trading library.

The current scope includes European Black-Scholes pricing, analytical Greeks, risk-neutral GBM simulation, Monte Carlo pricing for European and arithmetic-average Asian options, statistical simulation validation, and antithetic variance reduction.

Features such as calibration, implied volatility surfaces, stochastic-volatility models, American-option numerical methods, and production market-data integration are intentionally outside the current scope.