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
