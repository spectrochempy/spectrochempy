# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Extended tests for 2D plotting functions.

Goal:
-----
Increase *behavioral* coverage of plot2d.py by exercising:
- plot_stack / plot_image / plot_map code paths
- keyword-driven branches
- preference interactions
- coordinate / transpose logic
- label and color handling

These tests intentionally avoid pixel/image comparison and instead
assert that plotting completes without errors and returns matplotlib
objects where applicable.
"""

import pytest
import numpy as np
import matplotlib.colors

from spectrochempy import NDDataset, read_omnic, preferences as prefs, show


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ds2d():
    """Shared 2D dataset for plot2D tests (loaded once per module)."""
    path = "irdata/nh4y-activation.spg"
    return read_omnic(path)


def _simple_2d_dataset():
    """Synthetic 2D dataset to trigger edge cases."""
    data = np.arange(20.0).reshape(4, 5)
    return NDDataset(data)


# -----------------------------------------------------------------------------
# plot_stack
# -----------------------------------------------------------------------------
def test_plot_stack_basic(ds2d):
    ds2d.plot_stack()
    show()


def test_plot_stack_transposed(ds2d):
    ds2d.plot_stack(data_transposed=True)
    show()


def test_plot_stack_with_kwargs(ds2d):
    ds2d.plot_stack(offset=0.5, lw=0.5, color="k")
    show()


def test_plot_stack_grayscale_style(ds2d):
    ax = ds2d.plot(style="grayscale")
    lines = ax.get_lines()
    assert len(lines) > 0
    for line in lines:
        color = line.get_color()
        if isinstance(color, str):
            rgb = matplotlib.colors.to_rgb(color)
        else:
            rgb = color[:3]
        assert abs(rgb[0] - rgb[1]) < 0.01 and abs(rgb[1] - rgb[2]) < 0.01


# -----------------------------------------------------------------------------
# plot_image
# -----------------------------------------------------------------------------
def test_plot_image_basic(ds2d):
    ds2d.plot_image()
    show()


def test_plot_image_with_colorbar_and_style(ds2d):
    ds2d.plot_image(colorbar=True, style=["sans", "paper"], fontsize=9)
    show()


def test_plot_image_colormap_preference(ds2d):
    prefs.reset()
    prefs.image.cmap = "magma"
    ds2d.plot_image()
    show()


@pytest.mark.xfail(
    reason="plot_image does not support data_transposed=True "
    "(resume logic requires line-based plots)",
    strict=False,
)
def test_plot_image_transposed(ds2d):
    ds2d.plot_image(data_transposed=True)


# -----------------------------------------------------------------------------
# plot_map
# -----------------------------------------------------------------------------
def test_plot_map_basic(ds2d):
    ds2d.plot_map()
    show()


def test_plot_map_no_colorbar(ds2d):
    ds2d.plot_map(colorbar=False)
    show()


# -----------------------------------------------------------------------------
# plot (dispatcher)
# -----------------------------------------------------------------------------
def test_plot_dispatch_stack(ds2d):
    ds2d.plot(method="stack")
    show()


def test_plot_dispatch_image(ds2d):
    ds2d.plot(method="image")
    show()


def test_plot_dispatch_map(ds2d):
    ds2d.plot(method="map")
    show()


# -----------------------------------------------------------------------------
# Color / label / cycle logic
# -----------------------------------------------------------------------------
def test_plot_single_color(ds2d):
    ds2d.plot(color="red")
    show()


def test_plot_cycle_color(ds2d):
    ds2d.plot(cmap=None)
    show()


# -----------------------------------------------------------------------------
# Preferences interaction
# -----------------------------------------------------------------------------
def test_plot_with_modified_preferences(ds2d):
    prefs.reset()
    prefs.font.size = 9
    prefs.font.weight = "bold"
    prefs.axes.grid = True

    ds2d.plot()
    show()


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------
def test_plot_with_synthetic_dataset():
    ds = _simple_2d_dataset()
    ds.plot_image()
    show()


def test_plot_map_with_synthetic_dataset():
    ds = _simple_2d_dataset()
    ds.plot_map()
    show()


def test_plot_stack_with_synthetic_dataset():
    ds = _simple_2d_dataset()
    ds.plot_stack()
    show()


# -----------------------------------------------------------------------------
# Projectipns
# -----------------------------------------------------------------------------
#  with image plot
def test_plot_image_with_x_projection(ds2d):
    fig = ds2d.plot_image(show_projection_x=True)
    assert fig is not None


# Images  +    Projections X and Y
def test_plot_image_with_xy_projections(ds2d):
    fig = ds2d.plot_image(
        show_projection_x=True,
        show_projection_y=True,
    )
    assert fig is not None


# Projections + colorbar
def test_plot_image_with_xy_projections(ds2d):
    fig = ds2d.plot_image(
        show_projection_x=True,
        show_projection_y=True,
    )
    assert fig is not None


# Projections + stack + resume
def test_plot_image_with_projections_and_colorbar(ds2d):
    fig = ds2d.plot_image(
        show_projection_x=True,
        show_projection_y=True,
        colorbar=True,
    )
    assert fig is not None


@pytest.mark.xfail(
    reason="plot_image does not support data_transposed=True "
    "(resume logic requires line-based plots)",
    strict=False,
)
# Projections with transpose
def test_plot_image_transposed_with_projections(ds2d):
    fig = ds2d.plot_image(
        data_transposed=True,
        show_projection_x=True,
        show_projection_y=True,
    )
    assert fig is not None


# degenerate case: single line (auto projection)
def test_plot_image_single_row_projection(ds2d):
    ds = ds2d[0:1, :]
    fig = ds.plot_image(show_projection_x=True)
    assert fig is not None


# degenerate case: single columns
def test_plot_image_single_column_projection(ds2d):
    ds = ds2d[:, 0:1]
    fig = ds.plot_image(show_projection_y=True)
    assert fig is not None


# projections + explicit limits
def test_plot_image_with_projections_and_limits(ds2d):
    fig = ds2d.plot_image(
        show_projection_x=True,
        xlim=(ds2d.x.data.min(), ds2d.x.data.mean()),
    )
    assert fig is not None
