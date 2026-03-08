# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Contract tests for Plot1D - parametrized testing of methods and kwargs.

This module exercises dispatch and style branches in plot1d.py using
parametrized tests with synthetic datasets.
"""

import pytest
import numpy as np


@pytest.fixture
def synthetic_1d():
    """Create a synthetic 1D dataset for testing."""
    import spectrochempy as scp

    x = np.linspace(4000, 400, 64)
    y = np.sin(x / 100) * 0.5 + 0.5
    return scp.NDDataset(y, coords=[x], units="dimensionless", title="Test 1D")


def test_plot1d_default(synthetic_1d):
    """Test plot1d default behavior."""
    ax = synthetic_1d.plot(show=False)
    assert ax is not None


def test_plot1d_with_color(synthetic_1d):
    """Test plot1d with color."""
    ax = synthetic_1d.plot(color="red", show=False)
    assert ax is not None


def test_plot1d_with_linewidth(synthetic_1d):
    """Test plot1d with linewidth."""
    ax = synthetic_1d.plot(linewidth=2, show=False)
    assert ax is not None


def test_plot1d_with_linestyle(synthetic_1d):
    """Test plot1d with linestyle."""
    ax = synthetic_1d.plot(linestyle="--", show=False)
    assert ax is not None


def test_plot1d_with_marker(synthetic_1d):
    """Test plot1d with marker."""
    ax = synthetic_1d.plot(marker="o", show=False)
    assert ax is not None


def test_plot1d_with_label(synthetic_1d):
    """Test plot1d with label."""
    ax = synthetic_1d.plot(label="test label", show=False)
    assert ax is not None


def test_plot1d_with_title(synthetic_1d):
    """Test plot1d with custom title."""
    ax = synthetic_1d.plot(title="Custom Title", show=False)
    assert ax.get_title() == "Custom Title"


def test_plot1d_with_xlabel_ylabel(synthetic_1d):
    """Test plot1d with custom axis labels."""
    ax = synthetic_1d.plot(xlabel="X Axis", ylabel="Y Axis", show=False)
    assert ax.get_xlabel() == "X Axis"
    assert ax.get_ylabel() == "Y Axis"


def test_plot1d_with_xlim(synthetic_1d):
    """Test plot1d with axis limits."""
    ax = synthetic_1d.plot(xlim=(1000, 3000), show=False)
    assert ax is not None


def test_plot1d_with_ylim(synthetic_1d):
    """Test plot1d with y-axis limits."""
    ax = synthetic_1d.plot(ylim=(0, 1), show=False)
    assert ax is not None
