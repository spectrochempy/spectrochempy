# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Behavioral tests for the ``output`` contract of the plotting API.

``output`` is documented on ``dataset.plot()``, its geometry shortcuts, and
the composite plotters. These tests check the *observable* result: a real image
file appears at the requested path, it holds the whole figure rather than a
bare axes, and it is written before any display step. The ``savefig``
preferences are checked through the produced files, not through mocks.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy.application.preferences import preferences as prefs

# ======================================================================================
# Helpers
# ======================================================================================

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


def _files(directory):
    """Return the sorted names of the files present in *directory*."""
    return sorted(item.name for item in directory.iterdir() if item.is_file())


def _is_png(path):
    """Return True if *path* starts with the PNG signature."""
    return path.read_bytes()[: len(PNG_SIGNATURE)] == PNG_SIGNATURE


def _small_dpi():
    """Keep the written images cheap while staying a real render."""
    prefs.savefig_dpi = 72


def _corner_alpha(path):
    """Return the alpha value of the top-left pixel of the image *path*."""
    from PIL import Image

    return Image.open(path).convert("RGBA").getpixel((0, 0))[3]


@pytest.fixture(autouse=True)
def _restore_touched_preferences():
    """
    Restore the preferences these tests change.

    ``preferences.reset()`` keeps the values assigned during a session, so the
    ``savefig`` traits have to be restored explicitly to stay independent.
    """
    keys = (
        "savefig_dpi",
        "savefig_format",
        "savefig_facecolor",
        "savefig_edgecolor",
        "savefig_bbox",
        "savefig_pad_inches",
        "savefig_transparent",
    )
    saved = {key: getattr(prefs, key) for key in keys}
    saved_figsize = prefs.figure.figsize
    yield
    for key, value in saved.items():
        setattr(prefs, key, value)
    prefs.figure.figsize = saved_figsize


@pytest.fixture
def nd_1d():
    """Small deterministic 1D dataset."""
    return NDDataset([1.0, 2.0, 3.0, 4.0], title="Intensity", units="absorbance")


@pytest.fixture
def nd_2d():
    """Small deterministic 2D dataset."""
    data = np.arange(12, dtype=float).reshape(3, 4)
    return NDDataset(data, title="Absorbance", units="absorbance")


# ======================================================================================
# dataset.plot: the file is really written
# ======================================================================================


class TestDatasetPlotOutput:
    """``dataset.plot(output=...)`` writes the figure to disk."""

    def test_writes_png_file(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "spectrum.png"

        ax = nd_1d.plot(output=target, show=False)

        assert _files(tmp_path) == ["spectrum.png"]
        assert _is_png(target)
        assert target.stat().st_size > 0
        assert isinstance(ax, plt.Axes)

    def test_accepts_str_path(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "as_string.png"

        nd_1d.plot(output=str(target), show=False)

        assert _is_png(target)

    def test_format_follows_extension(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "spectrum.jpg"

        nd_1d.plot(output=target, show=False)

        assert target.read_bytes()[: len(JPEG_SIGNATURE)] == JPEG_SIGNATURE

    def test_suffixless_uses_default_format(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "no_extension"

        nd_1d.plot(output=target, show=False)

        # The file is written at the exact requested name, without inventing
        # an extension, and the configured default format is used.
        assert _files(tmp_path) == ["no_extension"]
        assert _is_png(target)

    def test_2d_output(self, nd_2d, tmp_path):
        _small_dpi()
        target = tmp_path / "map.png"

        nd_2d.plot(output=target, show=False)

        assert _is_png(target)

    def test_without_output_no_file_is_created(self, nd_1d, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _small_dpi()

        ax = nd_1d.plot(show=False)

        assert _files(tmp_path) == []
        assert isinstance(ax, plt.Axes)

    def test_output_is_written_before_display(self, nd_1d, tmp_path, monkeypatch):
        """The file must exist when the display step is reached."""
        _small_dpi()
        target = tmp_path / "ordered.png"
        seen = {}

        def _record_display():
            seen["exists"] = target.is_file()

        monkeypatch.setattr("spectrochempy.utils.mplutils.show", _record_display)

        nd_1d.plot(output=target, show=True)

        assert seen["exists"] is True

    def test_output_does_not_display_when_show_is_false(self, nd_1d, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "spectrochempy.utils.mplutils.show", lambda: calls.append(1)
        )

        nd_1d.plot(show=False)

        assert calls == []


# ======================================================================================
# the whole figure is saved, not a bare axes
# ======================================================================================


class TestSavedFigureContent:
    """The file holds the complete figure."""

    def test_saves_the_axes_figure_not_the_current_one(
        self, nd_1d, tmp_path, clean_figures
    ):
        """Saving follows ``ax.figure`` instead of the active figure."""
        _small_dpi()
        target = tmp_path / "foreign.png"

        foreign = plt.figure(figsize=(6, 2))
        foreign_ax = foreign.add_subplot(111)
        # Make another figure the active one, so the two cannot be confused.
        plt.figure(figsize=(8, 8))

        nd_1d.plot(ax=foreign_ax, output=target, show=False)

        from PIL import Image

        assert Image.open(target).size == (int(6 * 72), int(2 * 72))

    def test_colorbar_is_inside_the_file(self, nd_2d, tmp_path):
        _small_dpi()
        prefs.figure.figsize = (6, 3)
        target = tmp_path / "colorbar.png"

        nd_2d.plot(method="pen", colorbar=True, output=target, show=False)

        from PIL import Image

        # A bare-axes save would be narrower than the full figure width.
        assert Image.open(target).size == (int(6 * 72), int(3 * 72))

    def test_overlay_contains_the_legend(self, nd_1d, tmp_path):
        _small_dpi()
        with_legend = tmp_path / "with_legend.png"
        without_legend = tmp_path / "without_legend.png"

        scp.plot_multiple(
            [nd_1d, nd_1d * 2, nd_1d * 3],
            labels=["a", "b", "c"],
            legend="best",
            output=with_legend,
            show=False,
        )
        scp.plot_multiple(
            [nd_1d, nd_1d * 2, nd_1d * 3],
            labels=["a", "b", "c"],
            output=without_legend,
            show=False,
        )

        assert _files(tmp_path) == ["with_legend.png", "without_legend.png"]
        # The legend is drawn after the traces, so it must be in the file.
        assert with_legend.read_bytes() != without_legend.read_bytes()


# ======================================================================================
# plot_multiple: the single-dataset delegation keeps the lifecycle flags
# ======================================================================================


class TestPlotMultipleSingleDataset:
    """A single dataset is delegated without losing ``show``, ``clear``, ``output``."""

    def test_single_dataset_draws_one_trace(self, nd_1d):
        """The delegation plots the dataset, not the values it iterates over."""
        result = scp.plot_multiple(nd_1d, method="pen", show=False)

        assert len(result.lines) == 1
        assert list(result.lines[0].get_xdata()) == [0, 1, 2, 3]

    def test_show_false_does_not_display(self, nd_1d, monkeypatch):
        """``show=False`` must survive the delegation to ``dataset.plot()``."""
        calls = []
        monkeypatch.setattr(
            "spectrochempy.utils.mplutils.show", lambda: calls.append(1)
        )

        scp.plot_multiple(nd_1d, show=False)

        assert calls == []

    def test_show_true_displays_once(self, nd_1d, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "spectrochempy.utils.mplutils.show", lambda: calls.append(1)
        )

        scp.plot_multiple(nd_1d, show=True)

        assert len(calls) == 1

    def test_output_is_written_without_display(self, nd_1d, tmp_path, monkeypatch):
        _small_dpi()
        target = tmp_path / "single.png"
        calls = []
        monkeypatch.setattr(
            "spectrochempy.utils.mplutils.show", lambda: calls.append(1)
        )

        scp.plot_multiple(nd_1d, output=target, show=False)

        assert _is_png(target)
        assert calls == []

    def test_explicit_axes_are_reused(self, nd_1d):
        """An explicit ``ax`` is used as is, without creating another axes."""
        figure = plt.figure()
        ax = figure.add_subplot(111)

        result = scp.plot_multiple(nd_1d, ax=ax, clear=False, show=False)

        assert result is ax
        assert result.get_figure() is figure

    def test_clear_false_reuses_the_current_figure(self, nd_1d):
        """``clear=False`` is forwarded, so the current figure is reused."""
        figure = plt.figure()

        result = scp.plot_multiple(nd_1d, clear=False, show=False)

        assert result.get_figure() is figure

    def test_default_clear_creates_a_figure(self, nd_1d, tmp_path):
        """Without ``ax`` and with ``clear=True``, the delegation makes its own figure."""
        _small_dpi()
        target = tmp_path / "own_figure.png"
        previous = plt.figure()

        result = scp.plot_multiple(nd_1d, output=target, show=False)

        assert result.get_figure() is not previous
        assert _is_png(target)


# ======================================================================================
# public geometry shortcuts share the dataset.plot lifecycle
# ======================================================================================


class TestPublicShortcutOutput:
    """Public geometry shortcuts use the same lifecycle as ``dataset.plot``."""

    @pytest.mark.parametrize(
        ("shortcut", "fixture_name"),
        [
            ("plot_pen", "nd_1d"),
            ("plot_scatter", "nd_1d"),
            ("plot_scatter_pen", "nd_1d"),
            ("plot_bar", "nd_1d"),
            ("plot_lines", "nd_2d"),
            ("plot_contour", "nd_2d"),
            ("plot_contourf", "nd_2d"),
            ("plot_stack", "nd_2d"),
            ("plot_map", "nd_2d"),
            ("plot_image", "nd_2d"),
            ("plot_surface", "nd_2d"),
            ("plot_waterfall", "nd_2d"),
        ],
    )
    def test_standalone_shortcut_writes_a_real_file(
        self, shortcut, fixture_name, request, tmp_path
    ):
        """Every public geometry shortcut reaches the shared finalizer."""
        _small_dpi()
        target = tmp_path / f"{shortcut}.png"
        dataset = request.getfixturevalue(fixture_name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = getattr(scp, shortcut)(dataset, output=target, show=False)

        assert isinstance(result, plt.Axes)
        assert _is_png(target)

    @pytest.mark.parametrize(
        ("shortcut", "method", "fixture_name"),
        [
            ("plot_pen", "pen", "nd_1d"),
            ("plot_image", "image", "nd_2d"),
            ("plot_surface", "surface", "nd_2d"),
        ],
    )
    def test_bound_shortcut_matches_explicit_method(
        self, shortcut, method, fixture_name, request, tmp_path
    ):
        """The bound shortcut and explicit method both save and return axes."""
        _small_dpi()
        dataset = request.getfixturevalue(fixture_name)
        shortcut_target = tmp_path / f"{shortcut}.png"
        method_target = tmp_path / f"{method}.png"

        shortcut_ax = getattr(dataset, shortcut)(
            output=shortcut_target,
            show=False,
        )
        method_ax = dataset.plot(method=method, output=method_target, show=False)

        assert type(shortcut_ax) is type(method_ax)
        assert _is_png(shortcut_target)
        assert _is_png(method_target)

    @pytest.mark.parametrize(("show", "expected_calls"), [(False, 0), (True, 1)])
    def test_shortcut_display_step_runs_at_most_once(
        self, nd_1d, monkeypatch, show, expected_calls
    ):
        calls = []
        monkeypatch.setattr(
            "spectrochempy.utils.mplutils.show", lambda: calls.append(1)
        )

        nd_1d.plot_pen(show=show)

        assert len(calls) == expected_calls

    def test_shortcut_saves_the_explicit_axes_figure(
        self, nd_1d, tmp_path, clean_figures
    ):
        _small_dpi()
        target = tmp_path / "shortcut-foreign.png"
        foreign = plt.figure(figsize=(6, 2))
        foreign_ax = foreign.add_subplot(111)
        plt.figure(figsize=(8, 8))

        result = nd_1d.plot_pen(ax=foreign_ax, output=target, show=False)

        from PIL import Image

        assert result is foreign_ax
        assert Image.open(target).size == (int(6 * 72), int(2 * 72))

    def test_internal_renderer_still_only_draws(self, nd_1d, tmp_path):
        """The internal renderer stays free of save and display ownership."""
        from spectrochempy.plotting.plot1d import plot_1D

        target = tmp_path / "internal.png"
        result = plot_1D(nd_1d, method="pen", output=target)

        assert isinstance(result, plt.Axes)
        assert not target.exists()


# ======================================================================================
# composite and multi-panel plotters
# ======================================================================================


class TestCompositeOutput:
    """Composite plotters save once, for the whole figure."""

    def test_two_axes_layout_writes_one_file(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "baseline.png"

        ax1, ax2 = scp.plot_baseline(
            nd_1d,
            nd_1d * 0.9,
            nd_1d * 0.8,
            output=target,
            show=False,
        )

        assert _files(tmp_path) == ["baseline.png"]
        assert _is_png(target)
        assert ax1.figure is ax2.figure

    def test_multiplot_writes_a_single_file(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "grid.png"

        axes = scp.multiplot(
            [nd_1d, nd_1d * 2, nd_1d * 3, nd_1d * 4],
            output=target,
            show=False,
        )

        assert _files(tmp_path) == ["grid.png"]
        assert _is_png(target)
        assert len(axes) == 4

    def test_multiplot_panels_are_silent_until_the_end(self, nd_1d, monkeypatch):
        """Only the grid owns the display step, not each of its panels."""
        calls = []
        monkeypatch.setattr(
            "spectrochempy.utils.mplutils.show", lambda: calls.append(1)
        )

        scp.multiplot([nd_1d, nd_1d * 2], show=True)

        assert len(calls) == 1

    def test_plot_merit_rejects_output_for_several_figures(self):
        """One ``output`` cannot describe the per-index figures."""
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        rng = np.random.RandomState(42)
        pca = scp.PCA(n_components=5)
        pca.fit(NDDataset(rng.randn(10, 8)))

        X = NDDataset(rng.randn(10, 8))
        X_hat = NDDataset(rng.randn(3, 10, 8))

        with pytest.raises(ValueError, match="each index is rendered in its own"):
            plot_merit(pca, X=X, X_hat=X_hat, index=[0, 1], output="unused.png")


# ======================================================================================
# the savefig preferences drive the output
# ======================================================================================


class TestSavefigPreferences:
    """The existing ``savefig`` preferences are applied."""

    def test_dpi_preference_is_applied(self, nd_1d, tmp_path):
        target = tmp_path / "dpi.png"
        prefs.savefig_dpi = 120

        nd_1d.plot(output=target, show=False)

        from PIL import Image

        assert round(Image.open(target).info["dpi"][0]) == 120

    def test_transparent_preference_is_applied(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "transparent.png"
        prefs.savefig_transparent = True

        nd_1d.plot(output=target, show=False)

        assert _corner_alpha(target) == 0

    def test_default_background_is_opaque(self, nd_1d, tmp_path):
        _small_dpi()
        target = tmp_path / "opaque.png"

        nd_1d.plot(output=target, show=False)

        assert _corner_alpha(target) == 255


# ======================================================================================
# errors
# ======================================================================================


class TestSaveErrors:
    """Failures are reported, not swallowed."""

    def test_missing_directory_raises_with_the_path(self, nd_1d, tmp_path):
        target = tmp_path / "missing" / "figure.png"

        with pytest.raises(OSError, match="missing"):
            nd_1d.plot(output=target, show=False)

        # SpectroChemPy does not silently create the parent directory.
        assert not (tmp_path / "missing").exists()

    def test_unwritable_destination_raises(self, nd_1d, tmp_path):
        target = tmp_path  # a directory, not a file

        with pytest.raises(OSError):
            nd_1d.plot(output=target, show=False)

    def test_invalid_output_type_raises(self, nd_1d):
        with pytest.raises(TypeError, match="must be a str or a pathlib.Path"):
            nd_1d.plot(output=42, show=False)


# ======================================================================================
# shared helper
# ======================================================================================


class TestResolveSaveFigure:
    """The helper maps a plot result to its figure."""

    def test_figure_and_axes_resolve_to_their_figure(self):
        from spectrochempy.utils.mplutils import _resolve_save_figure

        figure, ax = plt.subplots()

        assert _resolve_save_figure(figure) is figure
        assert _resolve_save_figure(ax) is figure

    def test_two_axes_of_one_figure(self):
        from spectrochempy.utils.mplutils import _resolve_save_figure

        figure, (ax1, ax2) = plt.subplots(2)

        assert _resolve_save_figure((ax1, ax2)) is figure

    def test_axes_of_two_figures_are_rejected(self):
        from spectrochempy.utils.mplutils import _resolve_save_figure

        _, ax1 = plt.subplots()
        _, ax2 = plt.subplots()

        with pytest.raises(ValueError, match="different figures"):
            _resolve_save_figure([ax1, ax2])

    def test_non_figure_target_is_rejected(self):
        from spectrochempy.utils.mplutils import _resolve_save_figure

        with pytest.raises(TypeError, match="Cannot determine the figure"):
            _resolve_save_figure("not a figure")
