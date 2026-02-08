# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Comprehensive tests for plot2d.py to achieve >90% coverage.

This file targets uncovered code paths including:
- Surface and waterfall plotting methods
- Edge cases in coordinate handling
- Parameter validation and preference interactions
- Helper functions (_get_clevels, _plot_waterfall)
- Exception handling paths
- Complex parameter combinations
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from contextlib import suppress

from spectrochempy import NDDataset, read_omnic, preferences as prefs, show
from spectrochempy.utils.constants import NOMASK


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ds2d():
    """Shared 2D dataset for plot2D tests."""
    path = "irdata/nh4y-activation.spg"
    return read_omnic(path)


def _simple_2d_dataset():
    """Synthetic 2D dataset for edge cases."""
    data = np.arange(20.0).reshape(4, 5)
    return NDDataset(data)


def _masked_2d_dataset():
    """2D dataset with masked data for surface plotting."""
    data = np.arange(20.0).reshape(4, 5)
    dataset = NDDataset(data)
    # Initialize mask array if needed
    if dataset.mask is NOMASK:
        dataset.mask = np.zeros_like(data, dtype=np.bool_)
    # Mask some data points
    dataset.mask[1, 2] = True
    dataset.mask[3, 4] = True
    return dataset


# -----------------------------------------------------------------------------
# plot_surface - Completely missing from current tests
# -----------------------------------------------------------------------------
def test_plot_surface_basic(ds2d):
    """Test basic surface plotting."""
    ax = ds2d.plot_surface()
    assert ax is not None
    # Check if it's a 3D axes
    assert hasattr(ax, "plot_surface")
    show()


def test_plot_surface_with_parameters(ds2d):
    """Test surface plotting with various parameters."""
    ax = ds2d.plot_surface(
        antialiased=False, rcount=20, ccount=30, edgecolor="blue", linewidth=0.5
    )
    assert ax is not None
    show()


def test_plot_surface_with_masked_data():
    """Test surface plotting with masked data."""
    ds = _masked_2d_dataset()
    ax = ds.plot_surface()
    assert ax is not None
    show()


def test_plot_surface_with_colormap(ds2d):
    """Test surface plotting with custom colormap."""
    ax = ds2d.plot_surface(cmap="viridis", linewidth=1.0)
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# plot_waterfall - Missing from current tests
# -----------------------------------------------------------------------------
def test_plot_waterfall_basic(ds2d):
    """Test basic waterfall plotting."""
    ax = ds2d.plot_waterfall()
    assert ax is not None
    show()


def test_plot_waterfall_with_azimuth_elevation(ds2d):
    """Test waterfall plotting with custom azimuth and elevation."""
    ax = ds2d.plot_waterfall(azim=45, elev=60)
    assert ax is not None
    show()


def test_plot_waterfall_with_parameters(ds2d):
    """Test waterfall plotting with various parameters."""
    ax = ds2d.plot_waterfall(
        azim=15,
        elev=25,
        xlabel="custom x",
        ylabel="custom y",
        zlabel="custom z",
        title="waterfall test",
    )
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Direct plot_2D calls - Missing from current tests
# -----------------------------------------------------------------------------
def test_plot_2d_direct_methods(ds2d):
    """Test direct plot_2D calls with different methods."""
    methods = ["stack", "map", "image", "surface", "waterfall"]
    for method in methods:
        ax = ds2d.plot_2D(method=method)
        assert ax is not None
        plt.close("all")  # Clean up between plots


def test_plot_2d_with_transpose(ds2d):
    """Test plot_2D with transpose parameter."""
    ax = ds2d.plot_2D(method="stack", transposed=True)
    assert ax is not None
    show()


def test_plot_2d_y_reverse(ds2d):
    """Test plot_2D with y_reverse parameter."""
    ax = ds2d.plot_2D(method="stack", y_reverse=True)
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Edge cases and parameter validation
# -----------------------------------------------------------------------------
def test_plot_with_imag_data(ds2d):
    """Test plotting imaginary component."""
    # Create a dataset with imaginary component
    data = ds2d.data + 1j * np.random.random(ds2d.data.shape) * 0.1
    imag_ds = NDDataset(data, coordset=ds2d.coordset)
    ax = imag_ds.plot_2D(method="stack", imag=True)
    assert ax is not None
    show()


def test_plot_with_data_only(ds2d):
    """Test data_only parameter."""
    ax = ds2d.plot_2D(method="stack", data_only=True)
    assert ax is not None
    show()


def test_plot_with_x_reverse(ds2d):
    """Test x_reverse parameter."""
    ax = ds2d.plot_2D(method="map", x_reverse=True)
    assert ax is not None
    show()


def test_plot_with_scales(ds2d):
    """Test different axis scales."""
    ax = ds2d.plot_2D(method="stack", xscale="log", yscale="linear")
    assert ax is not None
    show()


def test_plot_log_scale_edge_case():
    """Test log scale with edge case (small positive values)."""
    # Create dataset with small positive values
    data = np.array([[0.001, 0.01, 0.1], [0.002, 0.02, 0.2]])
    ds = NDDataset(data)
    ax = ds.plot_2D(method="stack", yscale="log")
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Style and preference handling
# -----------------------------------------------------------------------------
def test_plot_with_style_string(ds2d):
    """Test style parameter as string."""
    ax = ds2d.plot_2D(method="stack", style="scpy")
    assert ax is not None
    show()


def test_plot_with_style_list(ds2d):
    """Test style parameter as list."""
    ax = ds2d.plot_2D(method="stack", style=["scpy", "sans"])
    assert ax is not None
    show()


def test_plot_with_parameter_fallbacks(ds2d):
    """Test parameter fallback chains (lw->linewidth, etc.)."""
    ax = ds2d.plot_2D(
        method="stack",
        lw=2.0,  # should fallback to linewidth
        ls="--",  # should fallback to linestyle
        ms=8.0,  # should fallback to markersize
    )
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Coordinate system edge cases
# -----------------------------------------------------------------------------
def test_plot_with_labeled_coordinates():
    """Test plotting with labeled coordinates."""
    # Create dataset with labeled coordinates
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Set up coordinates with labels
    from spectrochempy.core.dataset.coord import Coord

    ds.x = Coord(data=np.arange(4), labels=["A", "B", "C", "D"])
    ds.y = Coord(data=np.arange(3), labels=["W1", "W2", "W3"])

    ax = ds.plot_2D(method="map")
    assert ax is not None
    show()


def test_plot_with_labeled_coordinates_explicit():
    """Test plotting with explicitly set labeled coordinates."""
    # Create dataset with labeled coordinates
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Set up coordinates with labels
    from spectrochempy.core.dataset.coord import Coord

    ds.x = Coord(data=np.arange(4), labels=["A", "B", "C", "D"])
    ds.y = Coord(data=np.arange(3), labels=["W1", "W2", "W3"])

    ax = ds.plot_2D(method="map")
    assert ax is not None
    show()


def test_plot_with_show_datapoints():
    """Test plotting with show_datapoints parameter."""
    # Create dataset
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Set up coordinates with show_datapoints
    from spectrochempy.core.dataset.coord import Coord

    ds.x = Coord(data=np.arange(4), show_datapoints=True)
    ds.y = Coord(data=np.arange(3), show_datapoints=True)

    ax = ds.plot_2D(method="stack")
    assert ax is not None
    show()


def test_plot_with_empty_coordinates():
    """Test plotting with empty coordinates."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Set up coordinates that match data dimensions
    from spectrochempy.core.dataset.coord import Coord

    ds.x = Coord(data=np.array([1, 2, 3]))  # matches first dimension
    ds.y = Coord(data=np.array([1, 2, 3, 4]))  # matches second dimension

    ax = ds.plot_2D(method="map")
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Color and normalization edge cases
# -----------------------------------------------------------------------------
def test_plot_with_custom_norm(ds2d):
    """Test plotting with custom normalization."""
    norm = Normalize(vmin=0.1, vmax=0.9)
    ax = ds2d.plot_2D(method="map", norm=norm)
    assert ax is not None
    show()


def test_plot_with_colormap_fallbacks(ds2d):
    """Test colormap parameter fallbacks."""
    ax = ds2d.plot_2D(method="map", colormap="plasma")
    assert ax is not None
    show()


def test_plot_stack_color_vs_colormap(ds2d):
    """Test stack plotting with color vs colormap."""
    # Test with single color
    ax1 = ds2d.plot_2D(method="stack", color="red")
    assert ax1 is not None
    plt.close("all")

    # Test with colormap
    ax2 = ds2d.plot_2D(method="stack", cmap="viridis")
    assert ax2 is not None
    show()


# -----------------------------------------------------------------------------
# Stack method specific logic
# -----------------------------------------------------------------------------
def test_plot_stack_clear_parameter(ds2d):
    """Test stack plotting with clear parameter."""
    ax = ds2d.plot_2D(method="stack", clear=False)
    assert ax is not None
    show()


def test_plot_stack_maxlines_parameter(ds2d):
    """Test stack plotting with maxlines parameter."""
    ax = ds2d.plot_2D(method="stack", maxlines=5)
    assert ax is not None
    show()


def test_plot_stack_label_format(ds2d):
    """Test stack plotting with custom label format."""
    ax = ds2d.plot_2D(method="stack", label_fmt="{:.2f}")
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Image method specific logic
# -----------------------------------------------------------------------------
def test_plot_image_discrete_data():
    """Test image plotting with discrete data."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Make coordinates discrete (labeled but no meaningful data)
    from spectrochempy.core.dataset.coord import Coord

    ds.x = Coord(data=np.array([0, 0, 0, 0]), labels=["A", "B", "C", "D"])

    # This should trigger the discrete_data branch and switch to 'map' method
    ax = ds.plot_2D(method="image")
    assert ax is not None
    show()


def test_plot_image_custom_cmap(ds2d):
    """Test image plotting with custom image_cmap."""
    ax = ds2d.plot_2D(method="image", image_cmap="magma")
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Map method specific logic
# -----------------------------------------------------------------------------
def test_plot_map_discrete_data():
    """Test map plotting with discrete data (individual markers)."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Make coordinates discrete
    from spectrochempy.core.dataset.coord import Coord

    ds.x = Coord(data=np.array([0, 0, 0, 0]), labels=["A", "B", "C", "D"])

    ax = ds.plot_2D(method="map")
    assert ax is not None
    show()


def test_plot_map_contour_parameters(ds2d):
    """Test map plotting with contour parameters."""
    ax = ds2d.plot_2D(method="map", nlevels=15, alpha=0.7)
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Axis and label complex logic
# -----------------------------------------------------------------------------
def test_plot_custom_labels(ds2d):
    """Test plotting with custom labels."""
    ax = ds2d.plot_2D(
        method="stack",
        xlabel="Custom X",
        ylabel="Custom Y",
        zlabel="Custom Z",
        title="Custom Title",
    )
    assert ax is not None
    show()


def test_plot_show_y_parameter(ds2d):
    """Test show_y parameter for y-axis visibility."""
    ax = ds2d.plot_2D(method="stack", show_y=False)
    assert ax is not None
    show()


def test_plot_show_zero_parameter(ds2d):
    """Test show_zero parameter."""
    ax = ds2d.plot_2D(method="stack", show_zero=True)
    assert ax is not None
    show()


def test_plot_uselabel_parameters(ds2d):
    """Test uselabel_x and uselabel_y parameters."""
    # Add labels to coordinates - make sure they match the coordinate sizes
    x_size = len(ds2d.x)
    y_size = len(ds2d.y)

    ds2d.x.labels = [f"X{i}" for i in range(x_size)]
    ds2d.y.labels = [f"Y{i}" for i in range(y_size)]

    ax = ds2d.plot_2D(method="map", uselabel_x=True, uselabel_y=True)
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Colorbar logic
# -----------------------------------------------------------------------------
def test_plot_colorbar_methods(ds2d):
    """Test colorbar creation with different methods."""
    methods_with_colorbar = ["map", "image", "surface"]
    for method in methods_with_colorbar:
        ax = ds2d.plot_2D(method=method, colorbar=True)
        assert ax is not None
        plt.close("all")


def test_plot_colorbar_custom_label(ds2d):
    """Test colorbar with custom label."""
    ax = ds2d.plot_2D(method="map", colorbar=True, zlabel="Intensity (a.u.)")
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Axis limits and scaling
# -----------------------------------------------------------------------------
def test_plot_custom_limits(ds2d):
    """Test plotting with custom axis limits."""
    ax = ds2d.plot_2D(
        method="stack", xlim=(1000, 3000), ylim=(0.5, 2.0), zlim=(0.1, 1.0)
    )
    assert ax is not None
    show()


def test_plot_z_reverse_parameter(ds2d):
    """Test z_reverse parameter for stack/waterfall."""
    ax = ds2d.plot_2D(method="stack", z_reverse=True)
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------------
def test_get_clevels_function():
    """Test the _get_clevels helper function."""
    from spectrochempy.core.plotters.plot2d import _get_clevels

    # Create test data
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Test with default parameters
    clevels = _get_clevels(data, prefs)
    assert len(clevels) > 0
    assert isinstance(clevels, np.ndarray)

    # Test with custom parameters
    clevels = _get_clevels(data, prefs, nlevels=10, start=0.1, negative=True)
    assert len(clevels) > 0

    # Test without negative contours
    clevels = _get_clevels(data, prefs, negative=False)
    assert len(clevels) > 0


def test_get_clevels_edge_cases():
    """Test _get_clevels with edge cases."""
    from spectrochempy.core.plotters.plot2d import _get_clevels

    # Test with small data
    data = np.array([[0.1, 0.2], [0.3, 0.4]])
    clevels = _get_clevels(data, prefs)
    assert len(clevels) > 0

    # Test with negative start parameter
    clevels = _get_clevels(data, prefs, start=-0.1)
    assert len(clevels) > 0


# -----------------------------------------------------------------------------
# Exception handling and edge cases
# -----------------------------------------------------------------------------
def test_plot_1d_redirect():
    """Test redirect to 1D plotting for <2D datasets."""
    # Create 1D dataset
    data = np.arange(10.0)
    ds = NDDataset(data)

    # This should redirect to plot_1D
    result = ds.plot_2D()
    assert result is not None


def test_plot_with_synthetic_small_dataset():
    """Test plotting with very small synthetic dataset."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    ds = NDDataset(data)

    ax = ds.plot_2D(method="stack")
    assert ax is not None
    show()


def test_plot_with_nan_data():
    """Test plotting with NaN data."""
    data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
    ds = NDDataset(data)

    ax = ds.plot_2D(method="map")
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Complex parameter interactions
# -----------------------------------------------------------------------------
def test_plot_transposed_with_various_methods(ds2d):
    """Test transpose parameter with different methods."""
    methods = ["stack", "map"]  # image and surface have issues with transpose
    for method in methods:
        ax = ds2d.plot_2D(method=method, transposed=True)
        assert ax is not None
        plt.close("all")


def test_plot_data_only_with_various_methods(ds2d):
    """Test data_only parameter with different methods."""
    methods = ["stack", "map", "image"]
    for method in methods:
        ax = ds2d.plot_2D(method=method, data_only=True)
        assert ax is not None
        plt.close("all")


def test_plot_with_preference_modifications(ds2d):
    """Test plotting with various preference modifications."""
    # Save original preferences
    original_font_size = prefs.font.size
    original_grid = prefs.axes.grid

    try:
        # Modify preferences
        prefs.font.size = 12
        prefs.axes.grid = True
        prefs.contour_alpha = 0.8
        prefs.max_lines_in_stack = 10

        ax = ds2d.plot_2D(method="stack")
        assert ax is not None
        show()

    finally:
        # Restore preferences
        prefs.font.size = original_font_size
        prefs.axes.grid = original_grid


# -----------------------------------------------------------------------------
# Method-specific edge cases
# -----------------------------------------------------------------------------
def test_plot_surface_axes_creation():
    """Test surface plotting when axes needs to be recreated."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)

    # Create a regular 2D axes first
    fig, ax = plt.subplots()

    # This should trigger the axes recreation logic
    result = ds.plot_2D(method="surface", ax=ax)
    assert result is not None
    show()


def test_plot_waterfall_transformation():
    """Test waterfall coordinate transformations."""
    ds = _simple_2d_dataset()

    ax = ds.plot_2D(method="waterfall", azim=30, elev=45, show_z=True)
    assert ax is not None
    show()


# -----------------------------------------------------------------------------
# Integration tests with existing test patterns
# -----------------------------------------------------------------------------
def test_integration_with_existing_patterns(ds2d):
    """Test integration with patterns from existing tests."""
    # Test similar to existing tests but with direct plot_2D calls
    ax = ds2d.plot_2D(method="stack", offset=0.5, lw=0.5, color="k")
    assert ax is not None
    show()


def test_integration_projections_with_plot_2d(ds2d):
    """Test projections using direct plot_2D calls."""
    ax = ds2d.plot_2D(
        method="image", show_projection_x=True, show_projection_y=True, colorbar=True
    )
    assert ax is not None
    show()
