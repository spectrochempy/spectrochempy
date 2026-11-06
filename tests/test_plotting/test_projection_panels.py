# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Numerical, geometry and lifecycle contracts for 2D marginal panels."""

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture
def spectral_map():
    return scp.NDDataset(
        [[-2.0, 1.0, 3.0, 0.0], [4.0, -1.0, 2.0, 1.0], [1.0, 5.0, -3.0, 2.0]],
        coordset=[
            scp.Coord([1800.0, 1600.0, 1400.0], units="cm^-1"),
            scp.Coord([1800.0, 1700.0, 1600.0, 1500.0], units="cm^-1"),
        ],
        units="absorbance",
    )


@pytest.mark.parametrize("reducer", ["sum", "mean", "max"])
def test_projection_values_match_retained_dimensions(spectral_map, reducer):
    result = spectral_map.plot_contour(
        projections=True,
        projection_reducer=reducer,
        return_dict=True,
        show=False,
    )
    axes = result["axes"]
    assert set(axes) == {"main", "top", "right"}
    reduction = getattr(np, reducer)
    np.testing.assert_allclose(
        axes["top"].lines[0].get_ydata(), reduction(spectral_map.data, axis=0)
    )
    np.testing.assert_allclose(
        axes["right"].lines[0].get_xdata(), reduction(spectral_map.data, axis=1)
    )
    np.testing.assert_allclose(axes["top"].lines[0].get_xdata(), spectral_map.x.data)
    np.testing.assert_allclose(axes["right"].lines[0].get_ydata(), spectral_map.y.data)


def test_projection_dimension_names_survive_transposition(spectral_map):
    result = spectral_map.T.plot_contour(projections="x", return_dict=True, show=False)
    assert set(result["axes"]) == {"main", "right"}
    np.testing.assert_allclose(
        result["axes"]["right"].lines[0].get_xdata(),
        spectral_map.data.sum(axis=0),
    )


def test_projection_preferences_and_explicit_override(spectral_map):
    scp.preferences.show_projection_x = True
    result = spectral_map.plot_contour(return_dict=True, show=False)
    assert set(result["axes"]) == {"main", "top"}
    result = spectral_map.plot_contour(projections=False, return_dict=True, show=False)
    assert set(result["axes"]) == {"main"}
    assert spectral_map.plot(method="lines", show=False) is not None


def test_masked_values_are_excluded_from_projection(spectral_map):
    mask = np.zeros(spectral_map.shape, dtype=bool)
    mask[0, 0] = True
    spectral_map.mask = mask
    result = spectral_map.plot_contour(projections="x", return_dict=True, show=False)
    np.testing.assert_allclose(
        result["axes"]["top"].lines[0].get_ydata(),
        np.ma.sum(spectral_map.masked_data, axis=0),
    )


def _reference(units="m^-1"):
    return scp.NDDataset(
        [1.0, 2.0, 4.0, 3.0, 0.0],
        coordset=[
            scp.Coord([180000.0, 172500.0, 165000.0, 157500.0, 150000.0], units=units)
        ],
        units="absorbance",
        title="Reference absorbance",
    )


def test_external_reference_converts_units_without_resampling_or_mutation(spectral_map):
    ref = _reference()
    old_x, old_data = ref.x.data.copy(), ref.data.copy()
    original_map = spectral_map.data.copy()
    result = spectral_map.plot_contour(
        projections="x",
        projection_data={"x": [ref, ref.copy()]},
        projection_kwargs={
            "x": {"labels": ["Start", "End"], "color": ["red", "black"], "legend": True}
        },
        return_dict=True,
        show=False,
    )
    top = result["axes"]["top"]
    assert len(top.lines) == 2
    np.testing.assert_allclose(top.lines[0].get_xdata(), old_x / 100)
    np.testing.assert_allclose(top.lines[0].get_ydata(), old_data)
    assert [t.get_text() for t in top.get_legend().get_texts()] == ["Start", "End"]
    np.testing.assert_array_equal(ref.x.data, old_x)
    np.testing.assert_array_equal(ref.data, old_data)
    np.testing.assert_array_equal(spectral_map.data, original_map)


def test_external_2d_series_adds_one_curve_per_spectrum(spectral_map):
    result = spectral_map.plot_contour(
        projection_data={"x": spectral_map},
        return_dict=True,
        show=False,
    )
    assert len(result["axes"]["top"].lines) == spectral_map.shape[0]


@pytest.mark.parametrize(
    "options, error",
    [
        ({"projections": "z"}, ValueError),
        ({"projections": ("x", "x")}, ValueError),
        ({"projections": 42}, TypeError),
        ({"projections": "x", "projection_reducer": "median"}, ValueError),
        ({"projections": "x", "projection_size": 0}, ValueError),
        ({"projections": "x", "projection_size": "25%"}, ValueError),
        ({"projections": "x", "projection_pad": -1}, ValueError),
        ({"projections": False, "projection_data": {"x": []}}, ValueError),
        ({"projection_data": []}, TypeError),
    ],
)
def test_invalid_projection_options_are_rejected(spectral_map, options, error):
    with pytest.raises(error):
        spectral_map.plot_contour(show=False, **options)


def test_incompatible_reference_coordinate_units_are_rejected(spectral_map):
    with pytest.raises(ValueError, match="compatible units"):
        spectral_map.plot_contour(projection_data={"x": _reference("s")}, show=False)


def test_projection_layout_aligns_square_map_and_colorbar(spectral_map):
    result = spectral_map.plot_contour(
        projections=True,
        projection_size=0.3,
        colorbar=True,
        equal_aspect=True,
        return_dict=True,
        show=False,
    )
    result["fig"].canvas.draw()
    main, top, right = (result["axes"][key] for key in ("main", "top", "right"))
    bounds = main.get_position()
    np.testing.assert_allclose(
        [top.get_position().x0, top.get_position().x1], [bounds.x0, bounds.x1]
    )
    np.testing.assert_allclose(
        [right.get_position().y0, right.get_position().y1], [bounds.y0, bounds.y1]
    )
    assert top.get_position().y0 > bounds.y1
    assert right.get_position().x0 > bounds.x1
    np.testing.assert_allclose(top.get_xlim(), main.get_xlim())
    np.testing.assert_allclose(right.get_ylim(), main.get_ylim())
    assert right.get_position().x1 < main._scp_colorbar.ax.get_position().x0
    assert len(result["fig"].axes) == 4


def test_explicit_axes_reuses_marginals_for_overlay_and_replaces_for_redraw(
    spectral_map,
):
    _, ax = plt.subplots()
    result = spectral_map.plot_contour(
        ax=ax, projections="x", return_dict=True, show=False
    )
    top = result["axes"]["top"]
    spectral_map.plot_contour(ax=ax, projections="x", clear=False, show=False)
    assert len(ax.figure.axes) == 2
    assert len(top.lines) == 2
    spectral_map.plot_contour(ax=ax, projections="x", clear=True, show=False)
    assert len(ax.figure.axes) == 2
    assert top not in ax.figure.axes
    spectral_map.plot_contour(ax=ax, projections=False, clear=True, show=False)
    assert ax.figure.axes == [ax]


def test_signed_contour_styles_and_colorbar_are_consistent(spectral_map):
    ax = spectral_map.plot_contour(
        levels=[-2, -1, 1, 2],
        positive_color="red",
        negative_color="blue",
        negative_linestyle="--",
        positive_linestyle="-",
        colorbar=True,
        show=False,
    )
    contours = ax.collections[0]
    np.testing.assert_allclose(
        contours.get_edgecolors(),
        [colors.to_rgba(c) for c in ["blue", "blue", "red", "red"]],
    )
    assert contours.get_linestyles()[0][1] is not None
    assert contours.get_linestyles()[-1][1] is None
    np.testing.assert_allclose(
        ax._scp_colorbar.mappable.to_rgba(-1), colors.to_rgba("blue")
    )
    np.testing.assert_allclose(
        ax._scp_colorbar.mappable.to_rgba(1), colors.to_rgba("red")
    )


def test_multiplot_owns_export_after_all_reference_panels(
    spectral_map, monkeypatch, tmp_path
):
    from spectrochempy.utils import mplutils

    observed = []
    original = mplutils._save_figure

    def save(fig, output):
        observed.append(len(fig.axes))
        original(fig, output)

    monkeypatch.setattr(mplutils, "_save_figure", save)
    output = tmp_path / "panels.png"
    result = scp.multiplot(
        [spectral_map, spectral_map],
        ncol=2,
        method="contour",
        projections="x",
        panel_kwargs=[
            {"projection_data": {"x": spectral_map}},
            {"projection_reducer": "mean"},
        ],
        return_dict=True,
        output=output,
        show=False,
    )
    assert observed == [4]
    assert output.stat().st_size > 1000
    assert len(result["panel_axes"]["axe11"]["top"].lines) == 3
    assert len(result["panel_axes"]["axe12"]["top"].lines) == 1


def test_multiplot_accepts_mixed_dimensions_and_row_heights(spectral_map):
    result = scp.multiplot(
        [spectral_map[0].squeeze(), spectral_map],
        nrow=2,
        method=["pen", "contour"],
        height_ratios=[1, 4],
        return_dict=True,
        show=False,
    )
    result["fig"].canvas.draw()
    assert len(result["axes"]) == 2
    np.testing.assert_allclose(
        result["axes"][1].get_position().height
        / result["axes"][0].get_position().height,
        4,
    )


def test_legacy_dimension_preference_survives_transpose(spectral_map):
    scp.preferences.show_projection_x = True
    result = spectral_map.T.plot_contour(return_dict=True, show=False)
    assert set(result["axes"]) == {"main", "right"}


def test_nan_samples_follow_contour_masking(spectral_map):
    values = spectral_map.data.copy()
    values[0, 0] = np.nan
    spectral_map.data = values
    result = spectral_map.plot_contour(projections="x", return_dict=True, show=False)
    np.testing.assert_allclose(
        result["axes"]["top"].lines[0].get_ydata(), np.nansum(values, axis=0)
    )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["contour", "contourf"])
def test_signed_normalization_ignores_nonfinite_values(spectral_map, nonfinite, method):
    values = spectral_map.data.copy()
    values[0, 0] = nonfinite
    spectral_map.data = values
    ax = spectral_map.plot(
        method=method,
        positive_color="red",
        negative_color="blue",
        show=False,
    )
    assert ax.collections[0].norm.vmin == -5.0
    assert ax.collections[0].norm.vmax == 5.0


def test_reference_dimensions_extend_implicit_projection_preferences(spectral_map):
    scp.preferences.show_projection_x = True
    result = spectral_map.plot_contour(
        projection_data={"y": spectral_map[:, 0].squeeze()},
        return_dict=True,
        show=False,
    )
    assert set(result["axes"]) == {"main", "top", "right"}
    np.testing.assert_allclose(
        result["axes"]["right"].lines[0].get_xdata(),
        spectral_map.data[:, 0],
    )


@pytest.mark.parametrize("method", ["contour", "contourf"])
def test_fully_masked_signed_map_has_finite_fallback_range(spectral_map, method):
    spectral_map.mask = np.ones(spectral_map.shape, dtype=bool)
    ax = spectral_map.plot(
        method=method,
        levels=[-1, 0, 1],
        positive_color="red",
        show=False,
    )
    assert ax.collections[0].norm.vmin == -1.0
    assert ax.collections[0].norm.vmax == 1.0


def test_reversed_index_axis_projects_the_displayed_rows():
    dataset = scp.NDDataset(np.arange(12.0).reshape(3, 4))
    result = dataset.plot_contour(
        projections="y", y_reverse=True, return_dict=True, show=False
    )
    np.testing.assert_allclose(
        result["axes"]["right"].lines[0].get_xdata(), dataset.data.sum(axis=1)[::-1]
    )


def test_complex_signal_projects_the_requested_component(spectral_map):
    dataset = spectral_map.copy()
    dataset.data = spectral_map.data + 1j * (spectral_map.data + 10)
    result = dataset.plot_contour(
        projections="x", imag=True, return_dict=True, show=False
    )
    np.testing.assert_allclose(
        result["axes"]["top"].lines[0].get_ydata(), dataset.data.imag.sum(axis=0)
    )


def test_reference_signal_conversion_uses_dataset_semantics(spectral_map):
    absorbance = spectral_map[0].squeeze()
    transmission = absorbance.to("transmittance")
    result = spectral_map.plot_contour(
        projection_data={"x": [absorbance, transmission]},
        return_dict=True,
        show=False,
    )
    first, second = result["axes"]["top"].lines
    np.testing.assert_allclose(first.get_ydata(), second.get_ydata())


def test_disjoint_reference_range_is_rejected(spectral_map):
    ref = _reference()
    ref.x.data = ref.x.data + 1e6
    with pytest.raises(ValueError, match="do not overlap"):
        spectral_map.plot_contour(projection_data={"x": ref}, show=False)


def test_single_map_export_supports_named_axes(spectral_map, tmp_path):
    output = tmp_path / "single.png"
    result = spectral_map.plot_contour(
        projections=True, return_dict=True, output=output, show=False
    )
    assert len(result["fig"].axes) == 3
    assert output.stat().st_size > 1000


def test_single_dataset_multiplot_named_return(spectral_map):
    result = scp.multiplot(
        spectral_map, method="contour", projections="x", return_dict=True, show=False
    )
    assert set(result["panel_axes"]["axe11"]) == {"main", "top"}


@pytest.mark.parametrize(
    "panel_kwargs", [[{}], [{"show": True}, {}], [{"method": "pen"}, {}]]
)
def test_multiplot_validates_per_panel_options(spectral_map, panel_kwargs):
    with pytest.raises(ValueError, match="panel_kwargs"):
        scp.multiplot(
            [spectral_map, spectral_map], panel_kwargs=panel_kwargs, show=False
        )


def test_signed_filled_contours_preserve_zero_boundary(spectral_map):
    ax = spectral_map.plot_contourf(
        levels=[-2, -1, 1, 2],
        positive_color="red",
        negative_color="blue",
        show=False,
    )
    contours = ax.collections[0]
    np.testing.assert_array_equal(contours.levels, [-2, -1, 0, 1, 2])
    np.testing.assert_allclose(
        contours.get_facecolors(),
        [colors.to_rgba(c) for c in ["blue", "blue", "red", "red"]],
    )


def test_none_sign_colors_keep_default_palette(spectral_map):
    assert (
        spectral_map.plot_contour(positive_color=None, negative_color=None, show=False)
        is not None
    )


@pytest.mark.parametrize("grid", [False, True])
def test_panel_title_is_above_top_projection(spectral_map, grid):
    options = {
        "projections": "x",
        "projection_kwargs": {"x": {"title": "Projection"}},
        "show": False,
        "return_dict": True,
    }
    if grid:
        result = scp.multiplot(
            [spectral_map, spectral_map],
            ncol=2,
            labels=["Map A", "Map B"],
            method="contour",
            **options,
        )
        axes = result["panel_axes"]["axe11"]
    else:
        result = spectral_map.plot_contour(title="Map A", **options)
        axes = result["axes"]
    result["fig"].canvas.draw()
    assert axes["main"].get_title() == ""
    assert axes["top"].get_title() == "Map A\nProjection"
    renderer = result["fig"].canvas.get_renderer()
    assert axes["top"].title.get_window_extent(renderer).y0 > axes["top"].bbox.y1


def test_overlay_preserves_panel_titles_unless_explicitly_replaced(spectral_map):
    result = spectral_map.plot_contour(
        title="Map A",
        projections="x",
        projection_kwargs={"x": {"title": "Reference"}},
        return_dict=True,
        show=False,
    )
    main, top = result["axes"]["main"], result["axes"]["top"]
    spectral_map.plot_contour(ax=main, projections="x", clear=False, show=False)
    assert top.get_title() == "Map A\nReference"
    spectral_map.plot_contour(
        ax=main,
        title="Map B",
        projections="x",
        clear=False,
        show=False,
    )
    assert top.get_title() == "Map B\nReference"
    spectral_map.plot_contour(ax=main, title="Map C", clear=False, show=False)
    assert top.get_title() == "Map C\nReference"
    assert main.get_title() == ""
    spectral_map.plot_contour(ax=main, clear=False, show=False)
    assert top.get_title() == "Map C\nReference"
    spectral_map.plot_contour(
        ax=main,
        title="",
        projections="x",
        projection_kwargs={"x": {"title": ""}},
        clear=False,
        show=False,
    )
    assert top.get_title() == ""
    spectral_map.plot_contour(ax=main, projections="x", clear=True, show=False)
    assert main._scp_projection_axes["top"].get_title() == ""


def test_removing_projections_clears_the_previous_panel_title(spectral_map):
    result = spectral_map.plot_contour(
        title="Old",
        projections="x",
        return_dict=True,
        show=False,
    )
    main = result["axes"]["main"]
    spectral_map.plot_contour(ax=main, projections=False, clear=True, show=False)
    assert main.get_title() == ""
    spectral_map.plot_contour(ax=main, projections="x", clear=False, show=False)
    assert main._scp_projection_axes["top"].get_title() == ""


@pytest.mark.parametrize("selection", [None, True, "z"])
def test_map_with_nonstandard_dimensions_uses_only_existing_preferences(
    spectral_map, selection
):
    spectral_map.dims = ["z", "x"]
    options = {} if selection is None else {"projections": selection}
    result = spectral_map.plot_image(return_dict=True, show=False, **options)
    expected = {None: {"main"}, True: {"main", "top", "right"}, "z": {"main", "right"}}
    assert set(result["axes"]) == expected[selection]
    if selection is not None:
        np.testing.assert_allclose(
            result["axes"]["right"].lines[0].get_xdata(),
            spectral_map.data.sum(axis=1),
        )
