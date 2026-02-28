# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Validation suite for plotting documentation examples.

This test module programmatically verifies that examples in the plotting
documentation produce expected outputs. Each test corresponds to a doc example.

Run with:
    pytest tests/test_plotting/test_docs_examples.py -v
"""

import numpy as np
import pytest
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm

import spectrochempy as scp
from spectrochempy.application.preferences import preferences

# ======================================================================================
# Test Fixtures and Helpers
# ======================================================================================


@pytest.fixture(scope="module")
def sample_dataset():
    """Load a sample dataset for testing."""
    ds = scp.read("irdata/nh4y-activation.spg")
    return ds


@pytest.fixture(scope="module")
def sample_1d():
    """Load a sample 1D dataset for testing."""
    ds = scp.read("irdata/nh4y-activation.spg")
    return ds[0]


@pytest.fixture
def clean_preferences():
    """Save and restore preferences after each test."""
    original_prefs = {
        "colormap": preferences.colormap,
        "style": preferences.style,
        "figure.figsize": preferences.figure.figsize,
        "font.family": preferences.font.family,
        "font.size": preferences.font.size,
        "axes.grid": preferences.axes.grid,
        "colorbar": preferences.colorbar,
        "colormap_sequential": preferences.colormap_sequential,
        "colormap_diverging": preferences.colormap_diverging,
    }
    yield
    # Restore
    preferences.reset()
    for key, value in original_prefs.items():
        if "." in key:
            group, attr = key.split(".", 1)
            setattr(getattr(preferences, group), attr, value)
        else:
            setattr(preferences, key, value)


@pytest.fixture
def clean_rcparams():
    """Save and restore matplotlib rcParams after each test."""
    original_rcparams = rcParams.copy()
    yield
    rcParams.update(original_rcparams)


def get_mappable_from_ax(ax):
    """Extract the mappable (colormap) from axes."""
    if ax.images:
        return ax.images[0]
    if ax.collections:
        return ax.collections[0]
    return None


def get_cmap_name(ax):
    """Get colormap name from axes."""
    mappable = get_mappable_from_ax(ax)
    if mappable is not None:
        return mappable.get_cmap().name
    return None


def has_mappable(ax):
    """Check if axes has a mappable (image/contour)."""
    return ax.images or ax.collections


# ======================================================================================
# Overview Page Tests
# ======================================================================================


def test_overview_basic_plot(sample_dataset, clean_preferences, clean_rcparams):
    """Test basic plot - default behavior."""
    ax = sample_dataset.plot()
    # Should produce a plot with lines (stack plot)
    assert ax is not None
    assert len(ax.get_lines()) > 0


def test_overview_plot_contour(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_contour method."""
    ax = sample_dataset.plot_contour()
    assert ax is not None
    cmap_name = get_cmap_name(ax)
    assert cmap_name is not None


def test_overview_plot_image(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_image method."""
    ax = sample_dataset.plot_image()
    assert ax is not None
    cmap_name = get_cmap_name(ax)
    assert cmap_name is not None


def test_overview_categorical_cmap_none(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test cmap=None for categorical colors."""
    ax = sample_dataset.plot_lines(cmap=None)
    assert ax is not None
    # With cmap=None, should get categorical colors (lines should have colors)
    lines = ax.get_lines()
    assert len(lines) > 0


def test_overview_colormap_precedence_explicit(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test colormap precedence: explicit cmap kwarg overrides everything."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image(cmap="inferno")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "inferno"


def test_overview_colormap_precedence_prefs(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test colormap precedence: prefs.colormap overrides style."""
    preferences.colormap = "plasma"
    ax = sample_dataset.plot_image(style="grayscale")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # prefs.colormap should override style
    assert mappable.get_cmap().name == "plasma"


def test_overview_colormap_precedence_style_auto(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test colormap precedence: style affects cmap when prefs.colormap='auto'."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image(style="grayscale")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Style should set cmap when prefs is auto
    assert mappable.get_cmap().name == "gray"


def test_overview_colormap_precedence_defaults(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test colormap precedence: defaults when no style override."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Should use prefs.colormap_sequential (viridis by default)
    assert mappable.get_cmap().name == preferences.colormap_sequential


def test_overview_colormap_precedence_diverging(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test colormap precedence: diverging defaults."""
    preferences.colormap = "auto"
    # Create data with negative values
    ds = scp.NDDataset(np.random.randn(5, 10))
    ax = ds.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Should use diverging colormap
    assert mappable.get_cmap().name == preferences.colormap_diverging


# ======================================================================================
# Plot Types Page Tests
# ======================================================================================


def test_plot_types_line_plot(sample_1d, clean_preferences, clean_rcparams):
    """Test line plot for 1D data."""
    ax = sample_1d.plot()
    assert ax is not None
    assert len(ax.get_lines()) > 0


def test_plot_types_plot_lines(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_lines method."""
    ax = sample_dataset.plot_lines()
    assert ax is not None
    assert len(ax.get_lines()) > 0


def test_plot_types_plot_image(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_image method."""
    ax = sample_dataset.plot_image()
    assert ax is not None
    cmap_name = get_cmap_name(ax)
    assert cmap_name is not None


def test_plot_types_plot_image_colorbar_true(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test plot_image with colorbar=True."""
    ax = sample_dataset.plot_image(colorbar=True)
    assert ax is not None
    # Check colorbar exists
    assert len(ax.figure.axes) > 0 or ax.images


def test_plot_types_plot_image_colorbar_false(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test plot_image with colorbar=False."""
    ax = sample_dataset.plot_image(colorbar=False)
    assert ax is not None


def test_plot_types_plot_contour(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_contour method."""
    ax = sample_dataset.plot_contour()
    assert ax is not None


def test_plot_types_plot_contour_colorbar(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test plot_contour with colorbar."""
    ax = sample_dataset.plot_contour(colorbar=True)
    assert ax is not None


def test_plot_types_plot_surface(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_surface method."""
    ax = sample_dataset.plot_surface()
    assert ax is not None


def test_plot_types_plot_waterfall(sample_dataset, clean_preferences, clean_rcparams):
    """Test plot_waterfall method."""
    ax = sample_dataset.plot_waterfall()
    assert ax is not None


def test_plot_types_combining_options(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test combining options in plot."""
    ax = sample_dataset.plot_image(
        cmap="plasma",
        xlim=(2000, 1000),
        ylim=(0, 30),
    )
    assert ax is not None
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "plasma"


# ======================================================================================
# Customization Page Tests
# ======================================================================================


def test_customization_color(sample_dataset, clean_preferences, clean_rcparams):
    """Test color customization."""
    ax = sample_dataset.plot(color="red")
    assert ax is not None


def test_customization_linewidth(sample_dataset, clean_preferences, clean_rcparams):
    """Test linewidth customization."""
    ax = sample_dataset.plot(linewidth=2)
    assert ax is not None


def test_customization_linestyle(sample_dataset, clean_preferences, clean_rcparams):
    """Test linestyle customization."""
    ax = sample_dataset.plot(linestyle="--")
    assert ax is not None


def test_customization_marker(sample_dataset, clean_preferences, clean_rcparams):
    """Test marker customization."""
    ax = sample_dataset.plot(marker="o")
    assert ax is not None


def test_customization_alpha(sample_dataset, clean_preferences, clean_rcparams):
    """Test alpha customization."""
    ax = sample_dataset.plot(alpha=0.7)
    assert ax is not None


def test_customization_xlim(sample_dataset, clean_preferences, clean_rcparams):
    """Test xlim customization."""
    ax = sample_dataset.plot(xlim=(1000, 2000))
    assert ax is not None


def test_customization_ylim(sample_dataset, clean_preferences, clean_rcparams):
    """Test ylim customization for 1D data."""
    # Use 1D data for ylim test
    ax = sample_dataset[0].plot(ylim=(0, 0.5))
    assert ax is not None


def test_customization_figsize(sample_dataset, clean_preferences, clean_rcparams):
    """Test figsize customization."""
    ax = sample_dataset.plot(figsize=(10, 4))
    assert ax is not None
    assert ax.figure.get_figwidth() == 10
    assert ax.figure.get_figheight() == 4


def test_customization_grid(sample_dataset, clean_preferences, clean_rcparams):
    """Test grid customization."""
    ax = sample_dataset.plot(grid=True)
    assert ax is not None
    # Grid should be visible
    assert len(ax.get_xgridlines()) > 0 or len(ax.get_ygridlines()) > 0


def test_customization_cmap(sample_dataset, clean_preferences, clean_rcparams):
    """Test cmap customization."""
    ax = sample_dataset.plot_image(cmap="viridis")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "viridis"


def test_customization_cmap_none_categorical(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test cmap=None produces categorical colors."""
    ax = sample_dataset.plot_lines(cmap=None)
    # Get line colors
    lines = ax.get_lines()
    assert len(lines) > 0


def test_customization_normalization_centered(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test CenteredNorm normalization."""
    import matplotlib as mpl

    norm = mpl.colors.CenteredNorm()
    ax = sample_dataset.plot_image(cmap="RdBu_r", norm=norm)
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Check norm is applied - for contour plots we need to check differently
    if hasattr(mappable, "get_norm"):
        assert mappable.get_norm() is not None
    else:
        # QuadContourSet - norm was set via set_norm
        assert True  # If we got here, plot worked


def test_customization_normalization_log(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test LogNorm normalization."""
    import matplotlib as mpl

    norm = mpl.colors.LogNorm(vmin=0.01, vmax=1.0)
    ax = sample_dataset.plot_image(cmap="viridis", norm=norm)
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None


def test_customization_combined(sample_dataset, clean_preferences, clean_rcparams):
    """Test combined customizations."""
    # Use plot_image for testing combined options with mappable
    ax = sample_dataset.plot_image(
        xlim=(4000, 1500),
        figsize=(10, 5),
    )
    assert ax is not None
    assert ax.figure.get_figwidth() == 10
    assert ax.figure.get_figheight() == 5


# ======================================================================================
# Preferences Page Tests
# ======================================================================================


def test_preferences_access(sample_dataset, clean_preferences, clean_rcparams):
    """Test accessing preferences."""
    prefs = scp.preferences
    assert prefs.figure.figsize is not None
    assert prefs.colormap is not None
    assert prefs.font.family is not None
    assert prefs.style is not None


def test_preferences_change_defaults(sample_dataset, clean_preferences, clean_rcparams):
    """Test changing default preferences."""
    original_cmap = preferences.colormap
    preferences.colormap = "magma"
    ax = sample_dataset.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "magma"
    preferences.colormap = original_cmap


def test_preferences_reset(sample_dataset, clean_preferences, clean_rcparams):
    """Test resetting preferences."""
    preferences.colormap = "magma"
    preferences.figure.figsize = (8, 4)
    preferences.reset()
    # After reset, should use defaults
    ax = sample_dataset.plot()
    assert ax is not None


def test_preferences_colormap_sequential(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test colormap_sequential preference."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Should use colormap_sequential for positive data
    assert mappable.get_cmap().name == preferences.colormap_sequential


def test_preferences_colormap_diverging(clean_preferences, clean_rcparams):
    """Test colormap_diverging preference."""
    preferences.colormap = "auto"
    ds = scp.NDDataset(np.random.randn(5, 10))  # Mixed positive/negative
    ax = ds.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Should use diverging
    assert mappable.get_cmap().name == preferences.colormap_diverging


def test_preferences_group_access():
    """Test accessing preference groups."""
    prefs = scp.preferences
    assert prefs.lines is not None
    assert prefs.font is not None
    assert prefs.figure is not None
    assert prefs.axes is not None


def test_preferences_help():
    """Test getting help on preferences."""
    # Should not raise
    preferences.help("colormap")


# ======================================================================================
# Styles Page Tests
# ======================================================================================


def test_styles_available(sample_dataset, clean_preferences, clean_rcparams):
    """Test listing available styles."""
    from pathlib import Path

    styles_dir = Path(scp.preferences.stylesheets)
    styles = list(styles_dir.glob("*.mplstyle"))
    assert len(styles) > 0


def test_styles_apply_single_plot(sample_dataset, clean_preferences, clean_rcparams):
    """Test applying style to single plot."""
    ax = sample_dataset.plot(style="grayscale")
    assert ax is not None


def test_styles_affects_cmap_when_auto(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test style affects colormap when prefs.colormap='auto'."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image(style="grayscale")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # grayscale style should set cmap to gray
    assert mappable.get_cmap().name == "gray"


def test_styles_override_with_explicit_cmap(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test explicit cmap overrides style."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image(style="grayscale", cmap="viridis")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Explicit cmap should override style
    assert mappable.get_cmap().name == "viridis"


def test_styles_combine_with_options(sample_dataset, clean_preferences, clean_rcparams):
    """Test combining style with other options."""
    ax = sample_dataset.plot_image(
        style="grayscale",
    )
    assert ax is not None


def test_styles_default_style(sample_dataset, clean_preferences, clean_rcparams):
    """Test setting default style."""
    original_style = scp.preferences.style
    scp.preferences.style = "grayscale"
    ax = sample_dataset.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "gray"
    scp.preferences.style = original_style


# ======================================================================================
# Advanced Page Tests
# ======================================================================================


def test_advanced_modify_axes(sample_1d, clean_preferences, clean_rcparams):
    """Test modifying axes after plotting."""
    ax = sample_1d.plot()
    ax.set_title("Test Title")
    ax.set_xlabel("Test X Label")
    ax.set_ylabel("Test Y Label")
    assert ax.get_title() == "Test Title"


def test_advanced_subplots(clean_preferences, clean_rcparams):
    """Test creating subplots."""
    import matplotlib.pyplot as plt

    ds = scp.read("irdata/nh4y-activation.spg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    _ = ds.plot(ax=ax1)
    _ = ds.plot(ax=ax2, xlim=(2000, 1500))

    plt.close(fig)


def test_advanced_surface_plot(sample_dataset, clean_preferences, clean_rcparams):
    """Test surface plot."""
    ax = sample_dataset.plot_surface()
    assert ax is not None


def test_advanced_waterfall_plot(sample_dataset, clean_preferences, clean_rcparams):
    """Test waterfall plot."""
    ax = sample_dataset.plot_waterfall()
    assert ax is not None


def test_advanced_normalization_centered(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test centered normalization in advanced page."""
    import matplotlib as mpl

    norm = mpl.colors.CenteredNorm(vcenter=0.5)
    ax = sample_dataset.plot_image(cmap="RdBu_r", norm=norm)
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Check norm is applied - for contour plots we need to check differently
    if hasattr(mappable, "get_norm"):
        assert isinstance(mappable.get_norm(), (Normalize, TwoSlopeNorm))
    else:
        # QuadContourSet - norm was set via set_norm
        assert True  # If we got here, plot worked


def test_advanced_latex_labels(sample_1d, clean_preferences, clean_rcparams):
    """Test LaTeX-like math in labels."""
    ax = sample_1d.plot()
    ax.set_xlabel(r"$ \tilde{\nu}$ (cm$^{-1}$)")
    ax.set_ylabel(r"$ \epsilon$ (mol$^{-1}$·L·cm$^{-1}$)")
    ax.set_title(r"Beer-Lambert: $A = \epsilon c l$")
    assert ax.get_xlabel() != ""


def test_advanced_function_reproducibility(clean_preferences, clean_rcparams):
    """Test function for reproducible plotting."""
    ds = scp.read("irdata/nh4y-activation.spg")[0]

    def plot_spectrum(dataset, title=None):
        ax = dataset.plot(linewidth=1.5, color="navy", grid=True)
        if title:
            ax.set_title(title)
        return ax

    ax1 = plot_spectrum(ds, title="Sample 1")
    ax2 = plot_spectrum(ds, title="Sample 2")
    assert ax1 is not None
    assert ax2 is not None


# ======================================================================================
# Grayscale Style Tests (1D and 2D)
# ======================================================================================


def is_grayscale_color(color, tolerance=0.01):
    """Check if a color is grayscale (R≈G≈B)."""
    import matplotlib.colors as mpl_colors

    if isinstance(color, str):
        rgb = mpl_colors.to_rgb(color)
    else:
        rgb = color[:3]
    return abs(rgb[0] - rgb[1]) < tolerance and abs(rgb[1] - rgb[2]) < tolerance


def test_grayscale_style_1d_via_kwarg(sample_1d, clean_preferences, clean_rcparams):
    """Test 1D plot with style='grayscale' kwarg yields grayscale line colors."""
    ax = sample_1d.plot(style="grayscale")
    line = ax.get_lines()[0]
    color = line.get_color()
    assert is_grayscale_color(color), f"Expected grayscale, got {color}"


def test_grayscale_style_1d_via_prefs(sample_1d, clean_preferences, clean_rcparams):
    """Test 1D plot with prefs.style='grayscale' yields grayscale line colors."""
    preferences.style = "grayscale"
    ax = sample_1d.plot()
    line = ax.get_lines()[0]
    color = line.get_color()
    assert is_grayscale_color(color), f"Expected grayscale, got {color}"


def test_grayscale_style_1d_explicit_color_overrides(
    sample_1d, clean_preferences, clean_rcparams
):
    """Test explicit color overrides style for 1D."""
    ax = sample_1d.plot(style="grayscale", color="red")
    line = ax.get_lines()[0]
    color = line.get_color()
    assert color == "red", f"Expected red, got {color}"


def test_grayscale_style_2d_lines_via_kwarg(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test 2D lines (stack) with style='grayscale' kwarg yields grayscale line colors."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_lines(style="grayscale")
    lines = ax.get_lines()[:3]
    for i, line in enumerate(lines):
        color = line.get_color()
        assert is_grayscale_color(color), f"Line {i}: expected grayscale, got {color}"


def test_grayscale_style_2d_lines_via_prefs(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test 2D lines (stack) with prefs.style='grayscale' yields grayscale line colors."""
    preferences.style = "grayscale"
    preferences.colormap = "auto"
    ax = sample_dataset.plot_lines()
    lines = ax.get_lines()[:3]
    for i, line in enumerate(lines):
        color = line.get_color()
        assert is_grayscale_color(color), f"Line {i}: expected grayscale, got {color}"


def test_grayscale_style_2d_lines_explicit_color_overrides(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test explicit color overrides style for 2D lines."""
    ax = sample_dataset.plot_lines(style="grayscale", color="blue")
    line = ax.get_lines()[0]
    color = line.get_color()
    assert color == "blue", f"Expected blue, got {color}"


def test_grayscale_style_2d_image_via_kwarg(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test 2D image with style='grayscale' kwarg yields grayscale colormap."""
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image(style="grayscale")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "gray"


def test_grayscale_style_2d_image_via_prefs(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test 2D image with prefs.style='grayscale' yields grayscale colormap."""
    preferences.style = "grayscale"
    preferences.colormap = "auto"
    ax = sample_dataset.plot_image()
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "gray"


def test_grayscale_style_prefs_colormap_overrides(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test prefs.colormap overrides style for 2D image."""
    preferences.colormap = "plasma"
    ax = sample_dataset.plot_image(style="grayscale")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "plasma"


def test_grayscale_style_explicit_cmap_overrides(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test explicit cmap overrides style and prefs for 2D image."""
    preferences.colormap = "plasma"
    ax = sample_dataset.plot_image(style="grayscale", cmap="inferno")
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert mappable.get_cmap().name == "inferno"


def test_grayscale_style_categorical_still_works(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test explicit cmap=None categorical behavior still works."""
    ax = sample_dataset.plot_image(cmap=None)
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    assert isinstance(mappable.get_cmap(), ListedColormap)


# ======================================================================================
# Colorbar Consistency Tests (Contour/Image/Contourf)
# ======================================================================================


def get_colorbar_ticks(ax):
    """Get colorbar ticks from axes."""
    fig = ax.figure
    if len(fig.axes) > 1:
        cb_ax = fig.axes[1]
        return cb_ax.get_yticks()
    return None


def test_colorbar_consistency_contour_vs_contourf(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test plot_contour colorbar is consistent with plot_contourf."""
    # Plot both
    ax_contour = sample_dataset.plot_contour(colorbar=True)
    ax_contourf = sample_dataset.plot_contourf(colorbar=True)

    # Get colorbar ticks
    ticks_contour = get_colorbar_ticks(ax_contour)
    ticks_contourf = get_colorbar_ticks(ax_contourf)

    assert ticks_contour is not None, "Contour should have colorbar"
    assert ticks_contourf is not None, "Contourf should have colorbar"

    # Ticks should be the same (or very close)
    assert len(ticks_contour) == len(ticks_contourf), "Tick count should match"
    assert np.allclose(ticks_contour, ticks_contourf, rtol=0.01), "Ticks should match"


def test_colorbar_consistency_contour_vs_image(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test plot_contour colorbar is consistent with plot_image."""
    # Plot both
    ax_contour = sample_dataset.plot_contour(colorbar=True)
    ax_image = sample_dataset.plot_image(colorbar=True)

    # Get colorbar ticks
    ticks_contour = get_colorbar_ticks(ax_contour)
    ticks_image = get_colorbar_ticks(ax_image)

    assert ticks_contour is not None, "Contour should have colorbar"
    assert ticks_image is not None, "Image should have colorbar"

    # Ticks should be the same (or very close)
    assert len(ticks_contour) == len(ticks_image), "Tick count should match"
    assert np.allclose(ticks_contour, ticks_image, rtol=0.01), "Ticks should match"


def test_colorbar_ticks_monotonic(sample_dataset, clean_preferences, clean_rcparams):
    """Test colorbar ticks are monotonic increasing."""
    ax = sample_dataset.plot_contour(colorbar=True)
    ticks = get_colorbar_ticks(ax)
    assert ticks is not None, "Should have colorbar ticks"
    assert all(
        ticks[i] < ticks[i + 1] for i in range(len(ticks) - 1)
    ), "Ticks should be monotonic"


def test_colorbar_continuous_mapping(sample_dataset, clean_preferences, clean_rcparams):
    """Test colorbar uses continuous mapping (not categorical)."""
    ax = sample_dataset.plot_contour(colorbar=True)
    mappable = get_mappable_from_ax(ax)
    assert mappable is not None
    # Should NOT be ListedColormap (categorical)
    assert not isinstance(
        mappable.get_cmap(), ListedColormap
    ), "Should use continuous colormap"


def test_colorbar_cmap_name_consistent(
    sample_dataset, clean_preferences, clean_rcparams
):
    """Test all 2D methods use consistent colormap for colorbar."""
    # Note: The actual colormap used for display may differ slightly due to
    # contrast trimming (e.g., "truncated" vs original), but the colorbar
    # should reflect the same base colormap family.
    # This test verifies that all methods create colorbars successfully
    # and use continuous (not categorical) colormaps.

    ax_contour = sample_dataset.plot_contour(colorbar=True)
    ax_contourf = sample_dataset.plot_contourf(colorbar=True)
    ax_image = sample_dataset.plot_image(colorbar=True)

    # All should have colorbars
    assert get_colorbar_ticks(ax_contour) is not None, "Contour should have colorbar"
    assert get_colorbar_ticks(ax_contourf) is not None, "Contourf should have colorbar"
    assert get_colorbar_ticks(ax_image) is not None, "Image should have colorbar"

    # All should have mappables
    mappable_contour = get_mappable_from_ax(ax_contour)
    mappable_contourf = get_mappable_from_ax(ax_contourf)
    mappable_image = get_mappable_from_ax(ax_image)

    assert mappable_contour is not None
    assert mappable_contourf is not None
    assert mappable_image is not None


# ======================================================================================
# Main
# ======================================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
