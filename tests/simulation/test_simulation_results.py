import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from option_pricing.simulation.results import SimulationResult, AntitheticSimulationResult


@pytest.fixture
def valid_paths():
    return np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])


@pytest.fixture
def valid_time_grid():
    return np.array([0.0, 1.0, 2.0])


@pytest.fixture
def valid_pos_neg_paths():
    return {
        "positive_paths": np.array([[100.0, 105.0], [100.0, 110.0]]),
        "negative_paths": np.array([[100.0,  95.0], [100.0,  90.0]]),
        "time_grid": np.array([0.0, 1.0])
    }


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


def test_antithetic_simulation_result_valid():
    positive_paths = np.array([
        [100.0, 105.0, 110.0],
        [100.0, 102.0, 107.0],
    ])

    negative_paths = np.array([
        [100.0, 95.0, 90.0],
        [100.0, 98.0, 94.0],
    ])

    time_grid = np.array([0.0, 0.5, 1.0])

    result = AntitheticSimulationResult(
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        time_grid=time_grid,
    )

    np.testing.assert_array_equal(result.positive_paths, positive_paths)
    np.testing.assert_array_equal(result.negative_paths, negative_paths)
    np.testing.assert_array_equal(result.time_grid, time_grid)


def test_antithetic_simulation_result_rejects_mismatched_shapes():
    positive_paths = np.array([
        [100.0, 105.0],
        [100.0, 110.0],
    ])

    negative_paths = np.array([
        [100.0, 95.0],
    ])

    time_grid = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=positive_paths,
            negative_paths=negative_paths,
            time_grid=time_grid,
        )


def test_antithetic_simulation_result_arrays_are_read_only():
    result = AntitheticSimulationResult(
        positive_paths=np.array([[100.0, 105.0]]),
        negative_paths=np.array([[100.0, 95.0]]),
        time_grid=np.array([0.0, 1.0]),
    )

    with pytest.raises(ValueError):
        result.positive_paths[0, 1] = 999.0

    with pytest.raises(ValueError):
        result.negative_paths[0, 1] = 999.0

    with pytest.raises(ValueError):
        result.time_grid[1] = 2.0


def test_wrong_types(valid_pos_neg_paths):
    with pytest.raises(TypeError):
        AntitheticSimulationResult(
            positive_paths=[[100.0, 120.0]], 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=valid_pos_neg_paths["time_grid"]
        )
        
    with pytest.raises(TypeError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=[[100.0, 90.0]], 
            time_grid=valid_pos_neg_paths["time_grid"]
        )

    with pytest.raises(TypeError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=[0.0, 1.0]
        )


def test_non_numeric_arrays(valid_pos_neg_paths):
    string_array = np.array([["a", "b"], ["c", "d"]])

    with pytest.raises(TypeError):
        AntitheticSimulationResult(
            positive_paths=string_array, 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=valid_pos_neg_paths["time_grid"]
        )


def test_zero_paths(valid_pos_neg_paths):
    empty_paths = np.empty((0, 2))
    
    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=empty_paths, 
            negative_paths=empty_paths, 
            time_grid=valid_pos_neg_paths["time_grid"]
        )


def test_fewer_than_two_time_points():
    pos_paths = np.array([[1.0], [2.0]])
    neg_paths = np.array([[1.0], [2.0]])
    t_grid = np.array([0.0])
    
    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=pos_paths, 
            negative_paths=neg_paths, 
            time_grid=t_grid
        )


def test_time_grid_length_mismatch(valid_pos_neg_paths):
    wrong_grid = np.array([0.0, 1.0, 2.0])
    
    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=wrong_grid
        )


def test_time_grid_not_starting_at_zero(valid_pos_neg_paths):
    bad_start_grid = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=bad_start_grid
        )


def test_time_grid_not_strictly_increasing(valid_pos_neg_paths):
    flat_grid = np.array([0.0, 0.0])
    backwards_grid = np.array([0.0, -1.0])
    
    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=flat_grid
        )

    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=backwards_grid
        )


def test_nan_infinity(valid_pos_neg_paths):
    nan_paths = np.array([[1.0, np.nan], [1.0, 2.0]])
    inf_paths = np.array([[1.0, np.inf], [1.0, 2.0]])
    
    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=nan_paths, 
            negative_paths=valid_pos_neg_paths["negative_paths"], 
            time_grid=valid_pos_neg_paths["time_grid"]
        )

    with pytest.raises(ValueError):
        AntitheticSimulationResult(
            positive_paths=valid_pos_neg_paths["positive_paths"], 
            negative_paths=inf_paths, 
            time_grid=valid_pos_neg_paths["time_grid"]
        )