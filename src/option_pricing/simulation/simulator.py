import numpy as np

from option_pricing.simulation.process import StochasticProcess
from option_pricing.simulation.results import SimulationResult


class Simulator:
    """
    Simulates sample paths for a stochastic process.
    """

    def __init__(
        self,
        process: StochasticProcess,
        maturity: float,
        steps: int,
        n_paths: int,
        rng: np.random.Generator,
        ) -> None:

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be a StochasticProcess.")

        if maturity <= 0:
            raise ValueError("maturity must be strictly positive.")

        if type(steps) is not int:
            raise TypeError("steps must be an integer.")

        if steps < 1:
            raise ValueError("steps must be at least 1.")

        if type(n_paths) is not int:
            raise TypeError("n_paths must be an integer.")

        if n_paths < 1:
            raise ValueError("n_paths must be at least 1.")

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator.")

        self.process = process
        self.maturity = maturity
        self.steps = steps
        self.n_paths = n_paths
        self.rng = rng



    def run(self) -> SimulationResult:
        """Generate and return simulated process paths."""

        dt = self.maturity / self.steps

        time_grid = np.linspace(0.0, self.maturity, self.steps + 1)

        paths = np.empty((self.n_paths, self.steps + 1), dtype=float)

        paths[:, 0] = self.process.initial_value

        for step in range(self.steps):
            random_draws = self.rng.normal(size=self.n_paths)

            paths[:, step + 1] = self.process.step(
                current_value=paths[:, step],
                dt=dt,
                random_draw=random_draws
                )

        return SimulationResult(paths=paths, time_grid=time_grid)