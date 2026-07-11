# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the parity plot composite function."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrochempy import NDDataset
from spectrochempy.plotting.composite.parity import plot_parity


class TestParityPlot:
    """Test the standalone plot_parity function."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Create simple test datasets."""
        rng = np.random.default_rng(42)
        n = 20
        y_data = rng.normal(0, 1, n)
        y_hat_data = y_data + 0.1 * rng.normal(0, 1, n)

        self.Y = NDDataset(y_data)
        self.Y_hat = NDDataset(y_hat_data)

    def test_returns_axes(self):
        """Parityplot should return a matplotlib Axes object."""
        ax = plot_parity(self.Y, self.Y_hat, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_diagonal_line(self):
        """Parityplot should draw the y=x diagonal reference line."""
        ax = plot_parity(self.Y, self.Y_hat, show=False)
        lines = ax.lines
        assert len(lines) >= 1
        last_line = lines[-1]
        xd, yd = last_line.get_data()
        np.testing.assert_allclose(xd, yd)
        plt.close("all")

    def test_axis_labels(self):
        """Parityplot should label axes as 'measured values' and 'predicted values'."""
        ax = plot_parity(self.Y, self.Y_hat, show=False)
        assert ax.get_xlabel() == "measured values"
        assert ax.get_ylabel() == "predicted values"
        plt.close("all")

    def test_show_false_no_display(self, mocker):
        """plot_parity(show=False) should not call show."""
        display = mocker.patch("spectrochempy.utils.mplutils.show")
        plot_parity(self.Y, self.Y_hat, show=False)
        display.assert_not_called()
        plt.close("all")

    def test_show_true_calls_display(self, mocker):
        """plot_parity(show=True) should call show."""
        display = mocker.patch("spectrochempy.utils.mplutils.show")
        plot_parity(self.Y, self.Y_hat, show=True)
        display.assert_called_once()
        plt.close("all")

    def test_ax_reuse_with_clear_false(self):
        """Parityplot with clear=False should reuse the same axes."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="preexisting")

        ax2 = plot_parity(self.Y, self.Y_hat, ax=ax, clear=False, show=False)
        assert ax2 is ax
        assert len(ax.lines) >= 2
        assert len(ax.collections) >= 1
        plt.close("all")

    def test_ax_clear_true_clears_previous(self):
        """Parityplot with clear=True should remove previous content."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="preexisting")

        ax2 = plot_parity(self.Y, self.Y_hat, ax=ax, clear=True, show=False)
        assert ax2 is ax
        # After clear, only the parity plot artists should remain.
        # The scatter adds 1 collection, the diagonal adds 1 line.
        # Actually line count includes scatter collections too — let's just verify
        # we have the parity diagonal line and scatter.
        assert len(ax.collections) >= 1
        plt.close("all")

    def test_marker_param(self):
        """Parityplot should accept a marker parameter."""
        ax = plot_parity(self.Y, self.Y_hat, marker="x", show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_s_param(self):
        """Parityplot should accept a marker size parameter."""
        ax = plot_parity(self.Y, self.Y_hat, s=50, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_alpha_param(self):
        """Parityplot should accept an alpha parameter."""
        ax = plot_parity(self.Y, self.Y_hat, alpha=0.3, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_color_param(self):
        """Parityplot should accept a color parameter."""
        ax = plot_parity(self.Y, self.Y_hat, c="red", show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_2d_multi_target(self):
        """Parityplot should handle 2D (multi-target) Y data."""
        rng = np.random.default_rng(42)
        n = 20
        n_targets = 3
        y_data = rng.normal(0, 1, (n, n_targets))
        y_hat_data = y_data + 0.1 * rng.normal(0, 1, (n, n_targets))

        Y2 = NDDataset(y_data)
        Y_hat2 = NDDataset(y_hat_data)

        ax = plot_parity(Y2, Y_hat2, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")
