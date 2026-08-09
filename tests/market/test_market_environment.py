import pytest
from dataclasses import FrozenInstanceError

from option_pricing.instruments.underlying import Equity
from option_pricing.market.market_data import MarketData
from option_pricing.market.market_environment import MarketEnvironment


def test_valid_creation():
    aapl = Equity("AAPL", "NASDAQ")
    aapl_data = MarketData(
        spot=200,
        volatility=0.25,
        risk_free_rate=0.04
        )
    market = MarketEnvironment({aapl: aapl_data})
    assert market.data[aapl] == aapl_data


def test_multiple_assets():
    aapl = Equity("AAPL", "NASDAQ")
    aapl_data = MarketData(
        spot=200,
        volatility=0.25,
        risk_free_rate=0.04
        )

    msft = Equity("MSFT", "NASDAQ")
    msft_data = MarketData(
        spot=500,
        volatility=0.3,
        risk_free_rate=0.04
        )

    market = MarketEnvironment({aapl: aapl_data, msft: msft_data})

    assert market.data[aapl] == aapl_data
    assert market.data[msft] == msft_data


def test_empty_environment():
    with pytest.raises(ValueError, match="Market environment cannot be empty."):
        MarketEnvironment({})


def test_invalid_data_type():
    with pytest.raises(TypeError, match="Data must be provided as a Mapping."):
        MarketEnvironment([])


def test_invalid_underlying_key():
    aapl_data = MarketData(
            spot=200,
            volatility=0.25,
            risk_free_rate=0.04
            )
    with pytest.raises(TypeError, match="Keys must be Underlying objects."):
        MarketEnvironment(data={"AAPL": aapl_data})


def test_invalid_market_data_value():
    aapl = Equity("AAPL", "NASDAQ")
    with pytest.raises(TypeError, match="Values must be MarketData objects."):
        MarketEnvironment({aapl: 150})


def test_get_data_success():
    aapl = Equity("AAPL", "NASDAQ")
    aapl_data = MarketData(
            spot=200,
            volatility=0.25,
            risk_free_rate=0.04
            )
    market = MarketEnvironment({aapl: aapl_data})
    
    data = market.get_data(aapl)
    assert data == aapl_data


def test_get_data_missing_asset():
    aapl = Equity("AAPL", "NASDAQ")
    aapl_data = MarketData(
            spot=200,
            volatility=0.25,
            risk_free_rate=0.04
            )
    tsla = Equity("TSLA", "NASDAQ")

    market = MarketEnvironment({aapl: aapl_data})
    with pytest.raises(KeyError, match="No market data found for TSLA."):
        market.get_data(tsla)


def test_environment_attribute_immutability():
    aapl = Equity("AAPL", "NASDAQ")
    aapl_data = MarketData(
            spot=200,
            volatility=0.25,
            risk_free_rate=0.04
            )
    market = MarketEnvironment({aapl: aapl_data})    

    with pytest.raises(FrozenInstanceError):
        market.data = {}


def test_mapping_immutability():
    aapl = Equity("AAPL", "NASDAQ")
    aapl_data = MarketData(
        spot=200,
        volatility=0.25,
        risk_free_rate=0.04
        )

    msft = Equity("MSFT", "NASDAQ")
    msft_data = MarketData(
        spot=500,
        volatility=0.3,
        risk_free_rate=0.04
        )

    market = MarketEnvironment({aapl: aapl_data})    

    with pytest.raises(TypeError):
        market.data[msft] = msft_data

    with pytest.raises(TypeError):
        market.data[aapl] = msft_data