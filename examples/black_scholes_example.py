"""
Example: Pricing a European call option using the Black-Scholes model.

This script demonstrates the intended public workflow:
1. Define an underlying asset.
2. Define an option contract.
3. Define the market environment.
4. Apply a pricing model.
"""

from datetime import date

from option_pricing.instruments import (
    Equity,
    ExerciseStyle,
    Option,
    OptionType,
)
from option_pricing.market import MarketData, MarketEnvironment
from option_pricing.models import BlackScholesModel


def main():

    # 1. Define underlying asset
    apple = Equity(symbol="AAPL", exchange="NASDAQ")


    # 2. Define option contract
    call_option = Option(
        underlying=apple,
        strike=100,
        expiration_date=date(2027, 1, 1),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN
        )


    # 3. Define market environment
    market = MarketEnvironment(
        {
            apple: MarketData(
                spot=100,
                volatility=0.20,
                risk_free_rate=0.05,
                dividend_yield=0,
                )
            }, 
            valuation_date=date(2026, 1, 1)
            )


    # 4. Evaluate option
    model = BlackScholesModel()
    result = model.evaluate(call_option, market)


    # 5. Display pricing result
    print(f"Option price: {result.price:.4f}")
    print(f"Delta: {result.greeks.delta:.4f}")
    print(f"Gamma: {result.greeks.gamma:.6f}")
    print(f"Vega:  {result.greeks.vega:.4f}")
    print(f"Theta: {result.greeks.theta:.4f}")
    print(f"Rho:   {result.greeks.rho:.4f}")


if __name__ == "__main__":
    main()