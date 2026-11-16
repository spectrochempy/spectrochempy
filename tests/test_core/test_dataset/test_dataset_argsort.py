import numpy as np
import pytest

import spectrochempy as scp


@pytest.mark.parametrize(
    ("descend", "expected"),
    [
        (False, [4, 1, 3, 0, 7, 2, 5, 6]),
        (True, [7, 0, 1, 3, 4, 2, 5, 6]),
    ],
)
def test_argsort_is_stable_and_places_missing_values_last(descend, expected):
    dataset = scp.NDDataset(
        [3, 1, np.nan, 1, -np.inf, 2, np.nan, np.inf],
        mask=[False, False, False, False, False, True, False, False],
        coordset=[scp.Coord.arange(8)],
        units="ppm",
    )

    indices = dataset.argsort(descend=descend)

    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(dataset[indices].x.data, expected)
    np.testing.assert_array_equal(dataset.x.data, np.arange(8))
    np.testing.assert_array_equal(dataset[indices].mask[-3:], [False, True, False])


def test_argsort_can_rank_a_single_column_without_losing_row_indices():
    moisture = scp.NDDataset([[3], [1], [2]])
    indices = moisture.squeeze().argsort()

    np.testing.assert_array_equal(moisture[indices].data[:, 0], [1, 2, 3])


@pytest.mark.parametrize(("values", "indices"), [([], []), ([2], [0])])
def test_argsort_small_dataset(values, indices):
    result = scp.NDDataset(values).argsort()

    assert result.dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(result, indices)


@pytest.mark.parametrize("values", [[[2, 1]], [2 + 1j, 1 + 2j]])
def test_argsort_requires_one_dimensional_real_values(values):
    with pytest.raises(ValueError, match="one-dimensional real"):
        scp.NDDataset(values).argsort()
