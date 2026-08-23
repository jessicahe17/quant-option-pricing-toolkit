import numpy as np
from option_pricing.simulation.process import Numeric, StochasticProcess


class GBM(StochasticProcess):
    """
    Geometric Brownian Motion (GBM) stochastic process.

    The process follows
        dS_t = (r - q) S_t dt + sigma S_t dW_t

    and uses the exact GBM transition for discrete simulation.
    """

    def __init__(self, initial_value: float, volatility: float, rate: float, dividend_yield: float = 0.0) -> None:
        fields = {
            "initial_value": initial_value,
            "volatility": volatility,
            "rate": rate,
            "dividend_yield": dividend_yield,
        }

        for name, value in fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a numeric value.")

            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if initial_value <= 0:
            raise ValueError("initial_value must be positive.")

        if volatility < 0:
            raise ValueError("volatility must be non-negative.")

        if dividend_yield < 0:
            raise ValueError("dividend_yield must be non-negative.")

        self._initial_value = initial_value
        self.volatility = volatility
        self.rate = rate
        self.dividend_yield = dividend_yield


    @property
    def initial_value(self) -> float:
        """Return the initial value of the process."""
        return self._initial_value


    def step(self, current_value: Numeric, dt: float, random_draw: Numeric) -> Numeric:
        """Advance the GBM process by one time increment."""
        
        if dt <= 0:
            raise ValueError("Time increment dt must be strictly positive.")

        drift = self.rate - self.dividend_yield - 0.5 * self.volatility**2
        diffusion = self.volatility * np.sqrt(dt) * random_draw

        return current_value * np.exp(drift * dt + diffusion)