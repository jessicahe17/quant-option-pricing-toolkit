"""
Example: Pricing a European call option using the Black-Scholes model.

This script demonstrates the intended public workflow:
1. Define an underlying asset.
2. Define an option contract.
3. Define the market environment.
4. Apply a pricing model.
"""

from datetime import date

from option_pricing.instruments.underlying import Equity
from option_pricing.instruments.option import Option, OptionType, ExerciseStyle
from option_pricing.market.market_data import MarketData
from option_pricing.market.market_environment import MarketEnvironment
from option_pricing.models.black_scholes import BlackScholesModel


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


    # 4. Price option
    model = BlackScholesModel()
    price = model.price(call_option, market)

    print(f"Black-Scholes call price: {price:.4f}")


if __name__ == "__main__":
    main()