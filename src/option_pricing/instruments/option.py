from dataclasses import dataclass
from datetime import date
from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class Option:
    """Represents an option contract."""

    underlying: object
    strike: float
    expiration_date: date
    option_type: OptionType
    exercise_style: ExerciseStyle
    multiplier: float = 100

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError("Strike price must be positive.")

        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive.")

        if not isinstance(self.expiration_date, date):
            raise TypeError("Expiration date must be a datetime.date object.")

        if not isinstance(self.option_type, OptionType):
            raise TypeError("option_type must be an OptionType.")

        if not isinstance(self.exercise_style, ExerciseStyle):
            raise TypeError("exercise_style must be an ExerciseStyle.")


    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL


    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT


    def is_expired(self, as_of: date) -> bool:
        """
        Check whether the option has expired as of a given date.
        
        The option remains active on its expiration date and
        becomes expired only after that date.
        """
        return as_of > self.expiration_date