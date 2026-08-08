import pytest
from datetime import date
from dataclasses import FrozenInstanceError
from option_pricing.instruments.option import Option, OptionType, ExerciseStyle

def test_valid_option_creation():
    """Test valid creation and default multiplier."""

    aapl = object()

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
    aapl = object()
    expiration = date(2026, 11, 30)
    
    with pytest.raises(ValueError, match="Strike price must be positive."):
        Option(aapl, 0, expiration, OptionType.CALL, ExerciseStyle.AMERICAN)
        
    with pytest.raises(ValueError, match="Strike price must be positive."):
        Option(aapl, -50.5, expiration, OptionType.PUT, ExerciseStyle.EUROPEAN)


def test_invalid_multiplier():
    aapl = object()
    expiration = date(2026, 12, 30)

    
    with pytest.raises(ValueError, match="Multiplier must be positive."):
        Option(aapl, 120, expiration, OptionType.CALL, ExerciseStyle.AMERICAN, multiplier=0)
        
    with pytest.raises(ValueError, match="Multiplier must be positive."):
        Option(aapl, 150, expiration, OptionType.CALL, ExerciseStyle.EUROPEAN, multiplier=-10)


def test_invalid_option_type():
    aapl = object()

    with pytest.raises(TypeError, match="option_type must be an OptionType."):
        Option(aapl, 120, date(2026, 12, 30), "CALL", ExerciseStyle.AMERICAN)

    with pytest.raises(TypeError, match="option_type must be an OptionType."):
        Option(aapl, 150, date(2026, 12, 30), "PUT", ExerciseStyle.EUROPEAN)

    with pytest.raises(TypeError, match="option_type must be an OptionType."):
        Option(aapl, 150, date(2026, 12, 30), 1, ExerciseStyle.EUROPEAN)


def test_invalid_exercise_style():
    aapl = object()

    with pytest.raises(TypeError, match="exercise_style must be an ExerciseStyle."):
        Option(aapl, 140, date(2027, 8, 30), OptionType.CALL, "AMERICAN")

    with pytest.raises(TypeError, match="exercise_style must be an ExerciseStyle."):
        Option(aapl, 130, date(2027, 8, 30), OptionType.PUT, "european")


def test_immutability():
    aapl = object()
    opt = Option(aapl, 150, date(2026, 10, 30), OptionType.CALL, ExerciseStyle.AMERICAN)
    
    with pytest.raises(FrozenInstanceError):
        opt.strike = 160.0
        
    with pytest.raises(FrozenInstanceError):
        opt.multiplier = 200

    with pytest.raises(FrozenInstanceError):
        opt.expiration_date = date(2030, 10, 30)


def test_is_call_and_is_put():
    aapl = object()
    expiration = date(2026, 12, 31)
    
    call_opt = Option(aapl, 140, expiration, OptionType.CALL, ExerciseStyle.EUROPEAN)
    assert call_opt.is_call is True
    assert call_opt.is_put is False

    put_opt = Option(aapl, 200, expiration, OptionType.PUT, ExerciseStyle.AMERICAN)
    assert put_opt.is_call is False
    assert put_opt.is_put is True


def test_is_expired():
    aapl = object()
    expiration = date(2026, 10, 30)
    opt = Option(aapl, 160, expiration, OptionType.CALL, ExerciseStyle.AMERICAN)
    
    assert opt.is_expired(date(2026, 10, 29)) is False
    assert opt.is_expired(date(2026, 10, 30)) is False
    assert opt.is_expired(date(2026, 10, 31)) is True