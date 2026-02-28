# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for plot_baseline composite function."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.plotting.composite import plot_baseline
from spectrochempy.utils.exceptions import NotFittedError


def _make_synthetic_dataset_1d():
    x = scp.Coord(np.linspace(4000, 1000, 100), title="wavenumber", units="cm^-1")
    y = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.5
    return scp.NDDataset(y, coordset=[x])


def _make_synthetic_dataset_2d(n_traces=3):
    x = scp.Coord(np.linspace(4000, 1000, 100), title="wavenumber", units="cm^-1")
    y = scp.Coord(np.arange(n_traces), title="spectra")
    data_arr = np.zeros((n_traces, 100))
    for i in range(n_traces):
        data_arr[i] = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.5 + i * 0.1
    return scp.NDDataset(data_arr, coordset=[y, x])


class TestPlotBaseline:
    """Tests for plot_baseline composite function."""

    def test_returns_two_axes_and_two_axes_in_figure(self):
        orig = _make_synthetic_dataset_1d()
        baseline = orig * 0.2
        corrected = orig - baseline

        result = plot_baseline(orig, baseline, corrected, show=False)

        assert isinstance(result, tuple)
        assert len(result) == 2
        ax1, ax2 = result
        assert len(ax1.figure.axes) == 2
        plt.close()

    def test_sharex_true(self):
        orig = _make_synthetic_dataset_1d()
        baseline = orig * 0.2
        corrected = orig - baseline

        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)

        assert ax1.get_shared_x_axes().joined(ax1, ax2)
        plt.close()

    def test_zorder_baseline_in_front(self):
        orig = _make_synthetic_dataset_1d()
        baseline = orig * 0.2
        corrected = orig - baseline

        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)

        lines_top = ax1.get_lines()
        assert len(lines_top) == 2

        orig_line = lines_top[0]
        base_line = lines_top[1]

        assert base_line.get_zorder() > orig_line.get_zorder()
        plt.close()

    def test_corrected_colors_match_original(self):
        orig = _make_synthetic_dataset_2d(n_traces=3)
        baseline = orig * 0.2
        corrected = orig - baseline

        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)

        lines_top_orig = ax1.get_lines()[:3]
        lines_bottom = ax2.get_lines()

        for i in range(3):
            orig_color = lines_top_orig[i].get_color()
            corr_color = lines_bottom[i].get_color()
            if isinstance(orig_color, str) and isinstance(corr_color, str):
                assert orig_color == corr_color
            else:
                assert np.allclose(orig_color, corr_color)

        plt.close()

    def test_1d_single_trace(self):
        orig = _make_synthetic_dataset_1d()
        baseline = orig * 0.2
        corrected = orig - baseline

        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)

        lines_top = ax1.get_lines()
        lines_bottom = ax2.get_lines()

        assert len(lines_top) == 2
        assert len(lines_bottom) == 1

        plt.close()

    def test_rejects_ax_argument(self):
        orig = _make_synthetic_dataset_1d()
        baseline = orig * 0.2
        corrected = orig - baseline

        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="ax.*must be None"):
            plot_baseline(orig, baseline, corrected, ax=ax, show=False)

        plt.close()

    def test_shape_mismatch_raises(self):
        orig = _make_synthetic_dataset_2d(n_traces=3)
        baseline = orig * 0.2
        corrected_wrong = scp.NDDataset(np.zeros((2, 100)))

        with pytest.raises(ValueError, match="Shape mismatch"):
            plot_baseline(orig, baseline, corrected_wrong, show=False)

    def test_2d_multiple_traces(self):
        orig = _make_synthetic_dataset_2d(n_traces=5)
        baseline = orig * 0.2
        corrected = orig - baseline

        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)

        lines_top = ax1.get_lines()
        lines_bottom = ax2.get_lines()

        assert len(lines_top) == 10
        assert len(lines_bottom) == 5

        plt.close()


class TestBaselinePlotWrapper:
    """Tests for Baseline.plot() wrapper."""

    def test_baseline_plot_returns_two_axes(self):
        orig = _make_synthetic_dataset_1d()
        bl = scp.Baseline()
        bl.fit(orig)

        result = bl.plot(show=False)

        assert isinstance(result, tuple)
        assert len(result) == 2
        plt.close()

    def test_baseline_plot_not_fitted_raises(self):
        bl = scp.Baseline()

        with pytest.raises(NotFittedError):
            bl.plot(show=False)

    def test_baseline_plot_2d(self):
        orig = _make_synthetic_dataset_2d(n_traces=3)
        bl = scp.Baseline()
        bl.fit(orig)

        ax1, ax2 = bl.plot(show=False)

        assert len(ax1.figure.axes) == 2
        plt.close()


class TestPlotBaselineRegions:
    """Tests for region rendering in plot_baseline()."""

    def test_regions_rendered_on_top_axis(self):
        """Regions appear only on top axis, not bottom."""
        orig = _make_synthetic_dataset_1d()
        regions = [(3500, 3000), (1500, 1200)]

        ax1, ax2 = plot_baseline(
            orig,
            orig * 0.2,
            orig * 0.8,
            regions=regions,
            show_regions=True,
            show=False,
        )

        patches_top = [p for p in ax1.patches if hasattr(p, "get_width")]
        patches_bot = [p for p in ax2.patches if hasattr(p, "get_width")]

        assert len(patches_top) == len(regions)
        assert len(patches_bot) == 0

        plt.close()

    def test_regions_zorder_behind_lines(self):
        """Regions drawn behind all spectral lines (zorder=0)."""
        orig = _make_synthetic_dataset_1d()
        regions = [(3500, 3000)]

        ax1, ax2 = plot_baseline(
            orig,
            orig * 0.2,
            orig * 0.8,
            regions=regions,
            show_regions=True,
            show=False,
        )

        patches = [p for p in ax1.patches if hasattr(p, "get_width")]
        lines = ax1.get_lines()

        for patch in patches:
            assert patch.get_zorder() == 0
        for line in lines:
            assert line.get_zorder() >= 1

        plt.close()

    def test_regions_not_rendered_when_flag_false(self):
        """No regions rendered when show_regions=False."""
        orig = _make_synthetic_dataset_1d()
        regions = [(3500, 3000)]

        ax1, ax2 = plot_baseline(
            orig,
            orig * 0.2,
            orig * 0.8,
            regions=regions,
            show_regions=False,
            show=False,
        )

        patches = [p for p in ax1.patches if hasattr(p, "get_width")]
        assert len(patches) == 0

        plt.close()

    def test_regions_visible_with_inverted_axis(self):
        """Regions visible on inverted x-axis (IR convention)."""
        x = scp.Coord(np.linspace(4000, 1000, 100), title="wavenumber", units="cm^-1")
        y = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.5
        orig = scp.NDDataset(y, coordset=[x])
        regions = [(3500, 3000)]

        ax1, ax2 = plot_baseline(
            orig,
            orig * 0.2,
            orig * 0.8,
            regions=regions,
            show_regions=True,
            show=False,
        )

        xlim = ax1.get_xlim()
        assert xlim[0] > xlim[1]

        patches = [p for p in ax1.patches if hasattr(p, "get_width")]
        assert len(patches) == 1

        plt.close()

    def test_regions_no_regions_parameter(self):
        """No patches when regions=None even if show_regions=True."""
        orig = _make_synthetic_dataset_1d()

        ax1, ax2 = plot_baseline(
            orig,
            orig * 0.2,
            orig * 0.8,
            regions=None,
            show_regions=True,
            show=False,
        )

        patches = [p for p in ax1.patches if hasattr(p, "get_width")]
        assert len(patches) == 0

        plt.close()


class TestBaselinePlotWithRegions:
    """Tests for Baseline.plot() with show_regions parameter."""

    def test_baseline_plot_show_regions_true(self):
        """Baseline.plot(show_regions=True) renders regions."""
        orig = _make_synthetic_dataset_1d()
        bl = scp.Baseline()
        bl.ranges = [(3500, 3000), (1500, 1200)]
        bl.fit(orig)

        ax1, ax2 = bl.plot(show_regions=True, show=False)

        patches = [p for p in ax1.patches if hasattr(p, "get_width")]
        assert len(patches) >= 2

        plt.close()

    def test_baseline_plot_show_regions_false(self):
        """Baseline.plot(show_regions=False) does not render regions."""
        orig = _make_synthetic_dataset_1d()
        bl = scp.Baseline()
        bl.ranges = [(3500, 3000), (1500, 1200)]
        bl.fit(orig)

        ax1, ax2 = bl.plot(show_regions=False, show=False)

        patches = [p for p in ax1.patches if hasattr(p, "get_width")]
        assert len(patches) == 0

        plt.close()

    def test_show_regions_deprecated_warning(self):
        """show_regions() emits DeprecationWarning."""
        orig = _make_synthetic_dataset_1d()
        bl = scp.Baseline()
        bl.fit(orig)
        axes = bl.plot(show=False)

        with pytest.warns(DeprecationWarning, match="deprecated"):
            bl.show_regions(axes)

        plt.close()


class TestPlotClearFalseReusesAxis:
    """Tests for clear=False behavior - axes reuse."""

    def test_plot_clear_false_reuses_axis(self):
        """Multiple plot calls with clear=False reuse the same axis."""
        X = scp.NDDataset(np.random.randn(100))
        Y = X * 0.5
        Z = X - Y

        plt.close("all")

        ax1 = X.plot(label="a", show=False)
        ax2 = Y.plot(label="b", clear=False, show=False)
        ax3 = Z.plot(label="c", clear=False, show=False)

        assert ax1 is ax2 is ax3
        assert len(plt.get_fignums()) == 1

        plt.close("all")

    def test_plot_clear_false_without_existing_axis(self):
        """First call with clear=False creates a new figure."""
        plt.close("all")

        X = scp.NDDataset(np.random.randn(100))
        ax = X.plot(clear=False, show=False)

        assert len(plt.get_fignums()) == 1
        assert ax is not None

        plt.close("all")

    def test_plot_clear_true_creates_new_figure(self):
        """clear=True always creates a new figure."""
        X = scp.NDDataset(np.random.randn(100))

        plt.close("all")

        ax1 = X.plot(show=False)
        ax2 = X.plot(clear=True, show=False)

        assert ax1 is not ax2
        assert len(plt.get_fignums()) == 2

        plt.close("all")

    def test_plot_clear_false_then_clear_true(self):
        """clear=True creates new figure even after clear=False calls."""
        X = scp.NDDataset(np.random.randn(100))

        plt.close("all")

        ax1 = X.plot(show=False)
        ax2 = X.plot(clear=False, show=False)
        ax3 = X.plot(clear=True, show=False)

        assert ax1 is ax2
        assert ax2 is not ax3
        assert len(plt.get_fignums()) == 2

        plt.close("all")

    def test_detrend_example_pattern(self):
        """Detrend documentation example creates single figure."""
        x = scp.Coord(np.linspace(1000, 2000, 100), title="wavelength", units="nm")
        y = scp.Coord(np.arange(10), title="sample")
        data = np.random.randn(10, 100) + np.linspace(0, 1, 100)
        A = scp.NDDataset(data, coordset=[y, x])

        R = A[0]
        R1 = R.detrend()

        plt.close("all")

        ax1 = R.plot(label="original", show=False)
        ax2 = R1.plot(label="detrended", clear=False, show=False)
        ax3 = (R - R1).plot(label="trend", clear=False, show=False)

        assert ax1 is ax2 is ax3
        assert len(plt.get_fignums()) == 1

        plt.close("all")
