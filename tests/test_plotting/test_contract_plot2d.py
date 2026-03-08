# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Contract tests for Plot2D - parametrized testing of methods and kwargs.

This module exercises dispatch and style branches in plot2d.py using
parametrized tests with synthetic datasets.
"""

import pytest
import numpy as np


@pytest.fixture
def synthetic_2d():
    """Create a synthetic 2D dataset for testing."""
    import spectrochempy as scp

    x = np.linspace(4000, 400, 32)
    y = np.linspace(0, 60, 16)
    data = np.random.rand(16, 32)
    return scp.NDDataset(data, coords=[y, x], units="dimensionless", title="Test 2D")


def test_plot2d_image(synthetic_2d):
    """Test plot2d with image method."""
    ax = synthetic_2d.plot(method="image", show=False)
    assert ax is not None


def test_plot2d_map(synthetic_2d):
    """Test plot2d with map method."""
    ax = synthetic_2d.plot(method="map", show=False)
    assert ax is not None


def test_plot2d_contour(synthetic_2d):
    """Test plot2d with contour method."""
    ax = synthetic_2d.plot(method="contour", show=False)
    assert ax is not None


def test_plot2d_contourf(synthetic_2d):
    """Test plot2d with contourf method."""
    ax = synthetic_2d.plot(method="contourf", show=False)
    assert ax is not None


def test_plot2d_with_cmap(synthetic_2d):
    """Test plot2d with colormap."""
    ax = synthetic_2d.plot(cmap="viridis", show=False)
    assert ax is not None


def test_plot2d_with_colorbar_true(synthetic_2d):
    """Test plot2d with colorbar=True."""
    ax = synthetic_2d.plot(colorbar=True, show=False)
    assert ax is not None


def test_plot2d_with_colorbar_false(synthetic_2d):
    """Test plot2d with colorbar=False."""
    ax = synthetic_2d.plot(colorbar=False, show=False)
    assert ax is not None


def test_plot2d_transpose(synthetic_2d):
    """Test plot2d with transpose."""
    ax = synthetic_2d.plot(transpose=True, show=False)
    assert ax is not None


def test_plot2d_vmin_vmax(synthetic_2d):
    """Test plot2d with vmin/vmax."""
    ax = synthetic_2d.plot(vmin=0.2, vmax=0.8, show=False)
    assert ax is not None


def test_plot2d_custom_title(synthetic_2d):
    """Test plot2d with custom title."""
    ax = synthetic_2d.plot(title="Custom Title", show=False)
    assert ax.get_title() == "Custom Title"


def test_plot2d_custom_labels(synthetic_2d):
    """Test plot2d with custom labels."""
    ax = synthetic_2d.plot(xlabel="X Axis", ylabel="Y Axis", show=False)
    assert ax.get_xlabel() == "X Axis"
    assert ax.get_ylabel() == "Y Axis"


def test_plot2d_equal_aspect(synthetic_2d):
    """Test plot2d with equal_aspect."""
    ax = synthetic_2d.plot(equal_aspect=True, show=False)
    assert ax is not None


def test_plot2d_interpolation(synthetic_2d):
    """Test plot2d with interpolation."""
    ax = synthetic_2d.plot(interpolation="bilinear", show=False)
    assert ax is not None
