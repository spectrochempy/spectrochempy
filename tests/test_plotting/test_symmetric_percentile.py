"""Tests for symmetric percentile color limits."""

import matplotlib.pyplot as plt
import numpy as np

import spectrochempy as scp


def test_symmetric_percentile_respects_mask_and_preserves_data():
    """Exclude masked and non-finite samples from a display-only scale."""
    data = np.array([[1.0, -2, 3], [4, 100, np.nan], [2, 2, 2]])
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 1] = True
    dataset = scp.NDDataset(data, mask=mask)
    ax = dataset.plot_image(
        symmetric_percentile=50, cmap="RdBu_r", colorbar=True, show=False
    )
    try:
        assert ax.figure.axes[-1].get_ylim() == (-2.0, 2.0)
        np.testing.assert_array_equal(dataset.data, data)
        np.testing.assert_array_equal(dataset.mask, mask)
    finally:
        plt.close(ax.figure)


def test_symmetric_percentile_preserves_contour_crossing():
    dataset = scp.NDDataset(np.tile([0.0, 100.0, 2.0], (3, 1)))
    ax = dataset.plot_image(symmetric_percentile=50, nlevels=5, show=False)
    try:
        contour = ax.collections[0]
        assert contour.extend == "both"
        level_one = list(contour.levels).index(1.0) + 1
        xs = contour.get_paths()[level_one].vertices[:, 0]
        assert np.any(np.isclose(xs, 0.01))
    finally:
        plt.close(ax.figure)


def test_zero_percentile_stays_robust_to_an_isolated_extreme():
    data = np.zeros((3, 3))
    data[1, 1] = 100
    ax = scp.NDDataset(data).plot_image(symmetric_percentile=50, show=False)
    try:
        assert ax.collections[0].norm.vmax < 1
    finally:
        plt.close(ax.figure)
