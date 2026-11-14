"""Tests for linked upper x-axes on dataset plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrochempy as scp


@pytest.mark.parametrize(
    "method", ["pen", "scatter", "bar", "hist", "lines", "contour", "image"]
)
def test_secondary_x_tracks_primary_limits(method):
    """Convert the x limits when the primary axis changes."""
    data = np.arange(5)
    if method in {"lines", "contour", "image"}:
        data = np.arange(20).reshape(4, 5)
    dataset = scp.NDDataset(data)
    ax = dataset.plot(
        method=method,
        secondary_x=("Frequency (mHz)", lambda x: 5 * x, lambda f: f / 5),
        show=False,
    )
    try:
        (upper,) = ax.child_axes
        assert upper.get_xlabel() == "Frequency (mHz)"
        ax.set_xlim(3, 1)
        ax.figure.canvas.draw()
        np.testing.assert_allclose(upper.get_xlim(), (15, 5))
    finally:
        plt.close(ax.figure)


def test_secondary_x_on_overlay():
    """Add one upper axis to a plot with several traces."""
    dataset = scp.NDDataset(np.arange(5))
    ax = scp.plot_multiple(
        [dataset, dataset],
        method="pen",
        secondary_x=("Scaled x", lambda x: 5 * x, lambda x: x / 5),
        show=False,
    )
    try:
        assert len(ax.lines) == 2
        (upper,) = ax.child_axes
        assert upper.get_xlabel() == "Scaled x"
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("method", ["surface", "waterfall"])
def test_secondary_x_rejects_3d(method):
    """Reject upper axes on projected 3D plots."""
    dataset = scp.NDDataset(np.arange(20).reshape(4, 5))
    with pytest.raises(ValueError, match="2D Cartesian"):
        dataset.plot(
            method=method, secondary_x=("x", lambda x: x, lambda x: x), show=False
        )


@pytest.mark.parametrize("ndim", [1, 2])
def test_secondary_x_on_direct_renderer(ndim):
    dataset = scp.NDDataset(np.arange(9).reshape((3, 3) if ndim == 2 else (9,)))
    axis = ("Scaled x", lambda x: 2 * x, lambda x: x / 2)
    if ndim == 2:
        ax = dataset.plot_2D(method="contourf", secondary_x=axis)
    else:
        ax = dataset.plot_1D(method="pen", secondary_x=axis)
    try:
        (upper,) = ax.child_axes
        assert upper.get_xlabel() == "Scaled x"
    finally:
        plt.close(ax.figure)
