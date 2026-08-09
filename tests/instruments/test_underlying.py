import pytest
from dataclasses import FrozenInstanceError
from option_pricing.instruments.underlying import Equity


def test_valid_equity():
    eq = Equity("aapl", "nasdaq")
    
    assert eq.symbol == "AAPL"
    assert eq.exchange == "NASDAQ"

    eq_complex = Equity("BRK.B", "nyse")
    assert eq_complex.symbol == "BRK.B"
    assert eq_complex.exchange == "NYSE"


def test_invalid_symbols():
    # Test type error
    with pytest.raises(TypeError, match="Symbol must be a string."):
        Equity(123, "NASDAQ")
        
    # Test empty string
    with pytest.raises(ValueError, match="Symbol cannot be empty."):
        Equity("", "NASDAQ")
        
    # Test invalid spaces
    with pytest.raises(ValueError, match="Symbol contains invalid characters."):
        Equity(" AAPL", "NASDAQ")
        
    # Test invalid symbols
    with pytest.raises(ValueError, match="Symbol contains invalid characters."):
        Equity("AAPL!", "NASDAQ")


def test_invalid_exchange():
    # Test type error
    with pytest.raises(TypeError, match="Exchange must be a string."):
        Equity("AAPL", 1234)
        
    # Test empty string
    with pytest.raises(ValueError, match="Exchange cannot be empty."):
        Equity("AAPL", "")
        
    # Test leading whitespace
    with pytest.raises(ValueError, match="Exchange cannot contain leading or trailing whitespace."):
        Equity("AAPL", " NASDAQ")
        
    # Test trailing whitespace
    with pytest.raises(ValueError, match="Exchange cannot contain leading or trailing whitespace."):
        Equity("AAPL", "NASDAQ ")


def test_immutability():
    eq = Equity("AAPL", "NASDAQ")
    
    with pytest.raises(FrozenInstanceError):
        eq.symbol = "MSFT"
        
    with pytest.raises(FrozenInstanceError):
        eq.exchange = "NYSE"


def test_equalit_and_inequality():
    eq1 = Equity("aapl", "nasdaq")
    eq2 = Equity("AAPL", "NASDAQ")
    eq3 = Equity("MSFT", "NASDAQ")
    eq4 = Equity("AAPL", "NYSE")
    
    assert eq1 == eq2
    assert eq1 != eq3
    assert eq1 != eq4
    assert hash(eq1) == hash(eq2)