from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """
    Immutable result of a stochastic process simulation.

    Parameters
    ----------
    paths
        Simulated paths with shape
        (n_paths, n_time_points).

    time_grid
        Simulation time points with shape
        (n_time_points,).
    """

    paths: np.ndarray
    time_grid: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.paths, np.ndarray):
            raise TypeError("paths must be a NumPy array.")
        if not isinstance(self.time_grid, np.ndarray):
            raise TypeError("time_grid must be a NumPy array.")

        if not np.issubdtype(self.paths.dtype, np.number):
            raise TypeError("paths must have a numeric dtype.")
        if not np.issubdtype(self.time_grid.dtype, np.number):
            raise TypeError("time_grid must have a numeric dtype.")

        if self.paths.ndim != 2:
            raise ValueError("paths must be 2-dimensional.")
        if self.paths.shape[0] < 1:
            raise ValueError("paths must have at least one path.")
        if self.paths.shape[1] < 2:
            raise ValueError("paths must have at least two time points.")

        if self.time_grid.ndim != 1:
            raise ValueError("time_grid must be 1-dimensional.")
        if len(self.time_grid) != self.paths.shape[1]:
            raise ValueError("time_grid length must match the number of columns in paths.")
        
        if not np.all(np.isfinite(self.paths)):
            raise ValueError("paths must contain only finite values.")
        if not np.all(np.isfinite(self.time_grid)):
            raise ValueError("time_grid must contain only finite values.")
        
        if self.time_grid[0] != 0:
            raise ValueError("time_grid must start at 0.")
        if not np.all(np.diff(self.time_grid) > 0):
            raise ValueError("time_grid must be strictly increasing.")

        self.paths.flags.writeable = False
        self.time_grid.flags.writeable = False




@dataclass(frozen=True)
class MonteCarloPricingResult:
    """Result of a Monte Carlo option-pricing calculation."""

    price: float
    standard_error: float
    n_paths: int


    def __post_init__(self) -> None:
        if not np.isfinite(self.price):
            raise ValueError("price must be a finite number.")

        if not np.isfinite(self.standard_error):
            raise ValueError("standard_error must be a finite number.")
        if self.standard_error < 0:
            raise ValueError("standard_error must be non-negative.")

        if isinstance(self.n_paths, (bool, np.bool_)) or not isinstance(self.n_paths, (int, np.integer)):
            raise TypeError("n_paths must be an integer.")
            
        if self.n_paths < 2:
            raise ValueError("n_paths must be >= 2.")
