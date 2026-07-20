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

import numpy as np
import pytest


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


def test_plot1d_with_marker_none_does_not_crash(synthetic_1d):
    """
    marker=None is matplotlib's own idiom for "no marker" and must not crash.

    Regression test: `effective_marker = marker if marker.upper() != "AUTO"
    else None` called `.upper()` unconditionally, so passing `marker=None`
    (a legitimate, matplotlib-standard value, not just an unsupported edge
    case) raised `AttributeError: 'NoneType' object has no attribute
    'upper'` instead of simply plotting without a marker.
    """
    ax = synthetic_1d.plot(marker=None, show=False)
    assert ax is not None
    assert ax.lines[0].get_marker() in (None, "None")


def test_plot1d_with_linestyle_none_does_not_crash(synthetic_1d):
    """
    ls=None is matplotlib's own idiom for the default linestyle and must
    not crash.

    Regression test: the same unguarded `.upper()` pattern as the marker
    case above also existed for `ls` in the `pen=True` branch.
    """
    ax = synthetic_1d.plot(pen=True, ls=None, show=False)
    assert ax is not None
    assert ax.lines[0].get_linestyle() == "-"


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


def test_plot1d_alias_kwargs_normalize_to_canonical_artists(synthetic_1d):
    """Legacy line-style aliases should still drive the final Line2D artist."""
    ax = synthetic_1d.plot(
        c="red",
        lw=2.5,
        ls="--",
        marker="o",
        ms=7,
        mew=1.5,
        show=False,
    )

    line = ax.lines[0]
    assert line.get_color() == "red"
    assert line.get_linewidth() == pytest.approx(2.5)
    assert line.get_linestyle() == "--"
    assert line.get_marker() == "o"
    assert line.get_markersize() == pytest.approx(7)
    assert line.get_markeredgewidth() == pytest.approx(1.5)
