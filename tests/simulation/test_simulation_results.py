import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from option_pricing.simulation.results import SimulationResult


@pytest.fixture
def valid_paths():
    return np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])


@pytest.fixture
def valid_time_grid():
    return np.array([0.0, 1.0, 2.0])


def test_valid_simulation_result(valid_paths, valid_time_grid):
    result = SimulationResult(paths=valid_paths, time_grid=valid_time_grid)

    assert np.array_equal(result.paths, valid_paths)
    assert np.array_equal(result.time_grid, valid_time_grid)

    assert not result.paths.flags.writeable
    assert not result.time_grid.flags.writeable

    with pytest.raises(ValueError):
        result.paths[0, 0] = 10

    with pytest.raises(ValueError):
        result.time_grid[1] = 2.0

    with pytest.raises(FrozenInstanceError):
        result.paths = np.array([[1, 2], [3, 4]])


def test_paths_not_ndarray(valid_time_grid):
    with pytest.raises(TypeError, match="paths must be a NumPy array"):
        SimulationResult(paths=[[1.0, 2.0], [3.0, 4.0]], time_grid=valid_time_grid)


def test_time_grid_not_ndarray(valid_paths):
    with pytest.raises(TypeError, match="time_grid must be a NumPy array"):
        SimulationResult(paths=valid_paths, time_grid=[0.0, 1.0, 2.0])


def test_paths_non_numeric_dtype(valid_time_grid):
    paths = np.array([["a", "b", "c"]])
    with pytest.raises(TypeError, match="paths must have a numeric dtype"):
        SimulationResult(paths=paths, time_grid=valid_time_grid)


def test_time_grid_non_numeric_dtype(valid_paths):
    time_grid = np.array(["0", "1", "2"])
    with pytest.raises(TypeError, match="time_grid must have a numeric dtype"):
        SimulationResult(paths=valid_paths, time_grid=time_grid)


def test_paths_not_2d(valid_time_grid):
    paths_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="paths must be 2-dimensional"):
        SimulationResult(paths=paths_1d, time_grid=valid_time_grid)


def test_paths_empty(valid_time_grid):
    paths_empty = np.empty((0, 3))
    with pytest.raises(ValueError, match="paths must have at least one path"):
        SimulationResult(paths=paths_empty, time_grid=valid_time_grid)


def test_path_too_few_time_points():
    paths_short = np.array([[1], [2]])
    time_grid_short = np.array([0])
    with pytest.raises(ValueError, match="paths must have at least two time points"):
        SimulationResult(paths=paths_short, time_grid=time_grid_short)


def test_time_grid_not_1d(valid_paths):
    time_grid_2d = np.array([[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="time_grid must be 1-dimensional"):
        SimulationResult(paths=valid_paths, time_grid=time_grid_2d)


def test_length_mismatch(valid_paths):
    time_grid_short = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="time_grid length must match the number of columns in paths"):
        SimulationResult(paths=valid_paths, time_grid=time_grid_short)


def test_time_grid_does_not_start_at_zero(valid_paths):
    time_grid = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="time_grid must start at 0"):
        SimulationResult(paths=valid_paths, time_grid=time_grid)


def test_time_grid_not_strictly_increasing(valid_paths):
    with pytest.raises(ValueError, match="time_grid must be strictly increasing"):
        SimulationResult(paths=valid_paths, time_grid=np.array([0.0, 2.0, 1.0]))

    with pytest.raises(ValueError, match="time_grid must be strictly increasing"):
        SimulationResult(paths=valid_paths, time_grid=np.array([0.0, 1.0, 1.0]))


def test_paths_not_finite(valid_time_grid):
    paths_nan = np.array([[1.0, np.nan, 3.0]])
    with pytest.raises(ValueError, match="paths must contain only finite values"):
        SimulationResult(paths=paths_nan, time_grid=valid_time_grid)

    paths_inf = np.array([[1.0, 2.0, np.inf]])
    with pytest.raises(ValueError, match="paths must contain only finite values"):
        SimulationResult(paths=paths_inf, time_grid=valid_time_grid)


def test_time_grid_not_finite(valid_paths):
    time_grid_nan = np.array([0.0, 1.0, np.nan])
    with pytest.raises(ValueError, match="time_grid must contain only finite values"):
        SimulationResult(paths=valid_paths, time_grid=time_grid_nan)

    time_grid_inf = np.array([0.0, 1.0, np.inf])
    with pytest.raises(ValueError, match="time_grid must contain only finite values"):
        SimulationResult(paths=valid_paths, time_grid=time_grid_inf)