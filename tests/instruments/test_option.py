import pytest
from datetime import date
from dataclasses import FrozenInstanceError
from option_pricing.instruments.option import Option, OptionType, ExerciseStyle
from option_pricing.instruments.underlying import Equity

def test_valid_option_creation():
    """Test valid creation and default multiplier."""

    aapl = Equity("AAPL", "NASDAQ")

    opt = Option(
        underlying=aapl,
        strike=150,
        expiration_date=date(2026, 12, 30),
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN
    )

    assert isinstance(opt, Option)
    assert opt.underlying == aapl
    assert opt.strike == 150
    assert opt.expiration_date == date(2026, 12, 30)
    assert opt.option_type == OptionType.CALL
    assert opt.exercise_style == ExerciseStyle.AMERICAN
    assert opt.multiplier == 100
    


def test_invalid_strike():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 11, 30)
    
    with pytest.raises(ValueError, match="Strike price must be positive."):
        Option(aapl, 0, expiration, OptionType.CALL, ExerciseStyle.AMERICAN)
        
    with pytest.raises(ValueError, match="Strike price must be positive."):
        Option(aapl, -50.5, expiration, OptionType.PUT, ExerciseStyle.EUROPEAN)


def test_invalid_multiplier():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 12, 30)

    with pytest.raises(ValueError, match="Multiplier must be positive."):
        Option(aapl, 120, expiration, OptionType.CALL, ExerciseStyle.AMERICAN, multiplier=0)
        
    with pytest.raises(ValueError, match="Multiplier must be positive."):
        Option(aapl, 150, expiration, OptionType.CALL, ExerciseStyle.EUROPEAN, multiplier=-10)


def test_invalid_option_type():
    aapl = Equity("AAPL", "NASDAQ")

    with pytest.raises(TypeError, match="option_type must be an OptionType."):
        Option(aapl, 120, date(2026, 12, 30), "CALL", ExerciseStyle.AMERICAN)

    with pytest.raises(TypeError, match="option_type must be an OptionType."):
        Option(aapl, 150, date(2026, 12, 30), "PUT", ExerciseStyle.EUROPEAN)

    with pytest.raises(TypeError, match="option_type must be an OptionType."):
        Option(aapl, 150, date(2026, 12, 30), 1, ExerciseStyle.EUROPEAN)


def test_invalid_exercise_style():
    aapl = Equity("AAPL", "NASDAQ")

    with pytest.raises(TypeError, match="exercise_style must be an ExerciseStyle."):
        Option(aapl, 140, date(2027, 8, 30), OptionType.CALL, "AMERICAN")

    with pytest.raises(TypeError, match="exercise_style must be an ExerciseStyle."):
        Option(aapl, 130, date(2027, 8, 30), OptionType.PUT, "european")


def test_immutability():
    aapl = Equity("AAPL", "NASDAQ")
    opt = Option(aapl, 150, date(2026, 10, 30), OptionType.CALL, ExerciseStyle.AMERICAN)
    
    with pytest.raises(FrozenInstanceError):
        opt.strike = 160.0
        
    with pytest.raises(FrozenInstanceError):
        opt.multiplier = 200

    with pytest.raises(FrozenInstanceError):
        opt.expiration_date = date(2030, 10, 30)


def test_is_call_and_is_put():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 12, 31)
    
    call_opt = Option(aapl, 140, expiration, OptionType.CALL, ExerciseStyle.EUROPEAN)
    assert call_opt.is_call is True
    assert call_opt.is_put is False

    put_opt = Option(aapl, 200, expiration, OptionType.PUT, ExerciseStyle.AMERICAN)
    assert put_opt.is_call is False
    assert put_opt.is_put is True


def test_is_expired():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 10, 30)
    opt = Option(aapl, 160, expiration, OptionType.CALL, ExerciseStyle.AMERICAN)
    
    assert opt.is_expired(date(2026, 10, 29)) is False
    assert opt.is_expired(date(2026, 10, 30)) is False
    assert opt.is_expired(date(2026, 10, 31)) is True


def test_call_payoff():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 10, 30)
    call = Option(aapl, 100, expiration, OptionType.CALL, ExerciseStyle.AMERICAN)

    assert call.payoff(120) == 20
    assert call.payoff(101) == 1
    assert call.payoff(100) == 0
    assert call.payoff(99) == 0


def test_put_payoff():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 10, 30)
    put = Option(aapl, 100, expiration, OptionType.PUT, ExerciseStyle.AMERICAN)

    assert put.payoff(80) == 20
    assert put.payoff(99) == 1
    assert put.payoff(100) == 0
    assert put.payoff(101) == 0


def test_invalid_spot():
    aapl = Equity("AAPL", "NASDAQ")
    expiration = date(2026, 10, 30)
    opt = Option(aapl, 160, expiration, OptionType.CALL, ExerciseStyle.AMERICAN)

    with pytest.raises(ValueError, match="Spot price cannot be negative."):
        opt.payoff(-10)

    with pytest.raises(TypeError, match="Spot price must be numeric."):
        opt.payoff("100")