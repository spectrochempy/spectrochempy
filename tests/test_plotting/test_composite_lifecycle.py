# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Structural lifecycle contract tests for composite plot functions.

These tests verify that all composite plots adhere to the shared lifecycle
convention:

- ``ax=None`` → create new figure + axes via ``get_figure``
- ``ax=Axes`` + ``clear=True`` (default) → reuse, clear existing artists
- ``ax=Axes`` + ``clear=False`` → reuse without clearing
- ``show=False`` → return without calling ``mpl_show()``
- Return type is consistent (Axes or tuple of Axes)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy import NDDataset

# ======================================================================================
# Helpers
# ======================================================================================


def _draw_test_line(ax):
    """Draw a distinguishable line on *ax* so we can detect clearing."""
    ax.plot([0, 1], [0, 1], color="red", label="preexisting")
    return ax


def _has_preexisting_line(ax):
    """Return *True* if the preexisting red line is still present."""
    return any(
        hasattr(line, "get_color") and line.get_color() == "red" for line in ax.lines
    )


# ======================================================================================
# plot_score lifecycle
# ======================================================================================


class TestPlotScoreLifecycle:
    """Lifecycle contract for ``plot_score``."""

    @pytest.fixture
    def scores(self):
        """Small deterministic scores array (10 samples, 5 components)."""
        rng = np.random.RandomState(42)
        return NDDataset(rng.randn(10, 5))

    # -- ax=None creates a new figure+axes --
    def test_ax_none_creates_new_figure(self, scores):
        from spectrochempy.plotting.composite.plotscore import plot_score

        ax = plot_score(scores, show=False)
        assert isinstance(ax, plt.Axes)
        assert ax.figure is not None

    def test_ax_none_uses_get_figure_size_pref(self, scores):
        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotscore import plot_score

        orig = tuple(scp.preferences.figure.figsize)
        scp.preferences.figure.figsize = (6.0, 4.0)
        try:
            ax = plot_score(scores, show=False)
            w, h = ax.figure.get_size_inches()
            assert (w, h) == (6.0, 4.0)
        finally:
            scp.preferences.figure.figsize = orig

    # -- ax provided + clear=True (default) --
    def test_ax_clear_clears_artists(self, scores):
        from spectrochempy.plotting.composite.plotscore import plot_score

        _, ax = plt.subplots()
        _draw_test_line(ax)
        assert _has_preexisting_line(ax)

        ax = plot_score(scores, ax=ax, show=False)
        assert not _has_preexisting_line(ax)

    def test_ax_clear_default_is_true(self, scores):
        from spectrochempy.plotting.composite.plotscore import plot_score

        _, ax = plt.subplots()
        _draw_test_line(ax)
        n_before = len(ax.lines)

        ax = plot_score(scores, ax=ax, show=False)
        assert len(ax.lines) < n_before

    # -- ax provided + clear=False --
    def test_ax_no_clear_preserves_artists(self, scores):
        from spectrochempy.plotting.composite.plotscore import plot_score

        _, ax = plt.subplots()
        _draw_test_line(ax)
        n_before = len(ax.lines)

        ax = plot_score(scores, ax=ax, clear=False, show=False)
        assert len(ax.lines) >= n_before

    # -- show=False --
    def test_show_false_returns_axes(self, scores):
        from spectrochempy.plotting.composite.plotscore import plot_score

        ax = plot_score(scores, show=False)
        assert isinstance(ax, plt.Axes)
        assert ax.figure is not None

    # -- return type --
    def test_returns_axes(self, scores):
        from spectrochempy.plotting.composite.plotscore import plot_score

        ax = plot_score(scores, show=False)
        assert isinstance(ax, plt.Axes)


# ======================================================================================
# plot_scree lifecycle
# ======================================================================================


class TestPlotScreeLifecycle:
    """Lifecycle contract for ``plot_scree``."""

    @pytest.fixture
    def explained(self):
        return np.array([40.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0])

    def test_ax_none_creates_new_figure(self, explained):
        from spectrochempy.plotting.composite.plotscree import plot_scree

        ax = plot_scree(explained, show=False)
        assert isinstance(ax, plt.Axes)

    def test_ax_clear_clears_artists(self, explained):
        from spectrochempy.plotting.composite.plotscree import plot_scree

        _, ax = plt.subplots()
        _draw_test_line(ax)
        assert _has_preexisting_line(ax)

        ax = plot_scree(explained, ax=ax, show=False)
        assert not _has_preexisting_line(ax)

    def test_ax_no_clear_preserves_artists(self, explained):
        from spectrochempy.plotting.composite.plotscree import plot_scree

        _, ax = plt.subplots()
        _draw_test_line(ax)
        n_before = len(ax.lines)

        ax = plot_scree(explained, ax=ax, clear=False, show=False)
        assert len(ax.lines) >= n_before

    def test_show_false_returns_axes(self, explained):
        from spectrochempy.plotting.composite.plotscree import plot_scree

        ax = plot_scree(explained, show=False)
        assert isinstance(ax, plt.Axes)

    def test_returns_axes(self, explained):
        from spectrochempy.plotting.composite.plotscree import plot_scree

        ax = plot_scree(explained, show=False)
        assert isinstance(ax, plt.Axes)

    def test_twinx_creation(self, explained):
        from spectrochempy.plotting.composite.plotscree import plot_scree

        ax = plot_scree(explained, show=False)
        twin_axes = [c for c in ax.figure.axes if c is not ax]
        assert len(twin_axes) == 1


# ======================================================================================
# plot_compare lifecycle
# ======================================================================================


class TestPlotCompareLifecycle:
    """Lifecycle contract for ``plot_compare``."""

    def test_ax_none_creates_new_figure(self, sample_1d_dataset):
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        ax = plot_compare(X, X.copy(), show=False)
        assert isinstance(ax, plt.Axes)

    def test_ax_clear_clears_artists(self, sample_1d_dataset):
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        _, ax = plt.subplots()
        _draw_test_line(ax)
        assert _has_preexisting_line(ax)

        ax = plot_compare(X, X.copy(), ax=ax, show=False)
        assert not _has_preexisting_line(ax)

    def test_ax_no_clear_preserves_artists(self, sample_1d_dataset):
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        _, ax = plt.subplots()
        _draw_test_line(ax)
        n_before = len(ax.lines)

        ax = plot_compare(X, X.copy(), ax=ax, clear=False, show=False)
        assert len(ax.lines) >= n_before

    def test_show_false_returns_axes(self, sample_1d_dataset):
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        ax = plot_compare(sample_1d_dataset, sample_1d_dataset.copy(), show=False)
        assert isinstance(ax, plt.Axes)

    def test_returns_axes(self, sample_1d_dataset):
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        ax = plot_compare(sample_1d_dataset, sample_1d_dataset.copy(), show=False)
        assert isinstance(ax, plt.Axes)


# ======================================================================================
# plot_baseline lifecycle
# ======================================================================================


class TestPlotBaselineLifecycle:
    """
    Lifecycle contract for ``plot_baseline``.

    ``plot_baseline`` is the special case: it always creates its own
    two-axes figure and rejects explicit ``ax``.
    """

    @pytest.fixture
    def datasets(self):
        """Deterministic 1D datasets for baseline plotting."""
        x = scp.Coord(np.linspace(4000, 1000, 50), title="wavenumber", units="cm^-1")
        y = np.sin(np.linspace(0, 4 * np.pi, 50)) + 0.5
        orig = scp.NDDataset(y, coordset=[x])
        baseline = orig * 0.2
        corrected = orig - baseline
        return orig, baseline, corrected

    def test_ax_none_creates_new_figure(self, datasets):
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        orig, baseline, corrected = datasets
        result = plot_baseline(orig, baseline, corrected, show=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ax_rejected(self, datasets):
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        orig, baseline, corrected = datasets
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="must be None"):
            plot_baseline(orig, baseline, corrected, ax=ax, show=False)

    def test_returns_tuple_of_axes(self, datasets):
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        orig, baseline, corrected = datasets
        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)

    def test_two_panels_in_same_figure(self, datasets):
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        orig, baseline, corrected = datasets
        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)
        assert ax1.figure is ax2.figure
        assert len(ax1.figure.axes) == 2

    def test_sharex(self, datasets):
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        orig, baseline, corrected = datasets
        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)
        assert ax1.get_shared_x_axes().joined(ax1, ax2)

    def test_show_false_returns_axes(self, datasets):
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        orig, baseline, corrected = datasets
        ax1, ax2 = plot_baseline(orig, baseline, corrected, show=False)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)


# ======================================================================================
# plot_merit lifecycle
# ======================================================================================


class TestPlotMeritLifecycle:
    """Lifecycle contract for ``plot_merit``."""

    @pytest.fixture
    def trained_pca(self):
        rng = np.random.RandomState(42)
        X = NDDataset(rng.randn(20, 8))
        pca = scp.PCA(n_components=5)
        pca.fit(X)
        return pca

    def test_ax_none_creates_new_figure(self, trained_pca):
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        ax = plot_merit(trained_pca, show=False)
        assert isinstance(ax, plt.Axes)

    def test_ax_clear_clears_artists(self, trained_pca):
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        _, ax = plt.subplots()
        _draw_test_line(ax)
        assert _has_preexisting_line(ax)

        ax = plot_merit(trained_pca, ax=ax, show=False)
        assert not _has_preexisting_line(ax)

    def test_ax_no_clear_preserves_artists(self, trained_pca):
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        _, ax = plt.subplots()
        _draw_test_line(ax)
        n_before = len(ax.lines)

        ax = plot_merit(trained_pca, ax=ax, clear=False, show=False)
        assert len(ax.lines) >= n_before

    def test_multiparam_branch_ax_none_creates_figure(self, trained_pca):
        from spectrochempy import NDDataset
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        rng = np.random.RandomState(42)
        X = NDDataset(rng.randn(10, 8))
        X_hat = NDDataset(rng.randn(3, 10, 8))
        ax = plot_merit(trained_pca, X=X, X_hat=X_hat, show=False)
        assert isinstance(ax, plt.Axes)

    def test_multiparam_branch_clear_clears(self, trained_pca):
        from spectrochempy import NDDataset
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        rng = np.random.RandomState(42)
        X = NDDataset(rng.randn(10, 8))
        X_hat = NDDataset(rng.randn(3, 10, 8))
        _, ax = plt.subplots()
        _draw_test_line(ax)
        ax = plot_merit(trained_pca, X=X, X_hat=X_hat, ax=ax, show=False)
        assert not _has_preexisting_line(ax)

    def test_multiparam_branch_no_clear_preserves(self, trained_pca):
        from spectrochempy import NDDataset
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        rng = np.random.RandomState(42)
        X = NDDataset(rng.randn(10, 8))
        X_hat = NDDataset(rng.randn(3, 10, 8))
        _, ax = plt.subplots()
        _draw_test_line(ax)
        ax = plot_merit(trained_pca, X=X, X_hat=X_hat, ax=ax, clear=False, show=False)
        assert _has_preexisting_line(ax)

    def test_index_single_delegates_to_compare(self, trained_pca):
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        ax = plot_merit(trained_pca, index=0, show=False)
        assert isinstance(ax, plt.Axes)

    def test_index_iterable_returns_list(self, trained_pca):
        from spectrochempy import NDDataset
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        rng = np.random.RandomState(42)
        X = NDDataset(rng.randn(10, 8))
        X_hat = NDDataset(rng.randn(3, 10, 8))
        axes = plot_merit(trained_pca, X=X, X_hat=X_hat, index=[0, 1], show=False)
        assert isinstance(axes, list)
        assert all(isinstance(a, plt.Axes) for a in axes)
        assert len(axes) == 2

    def test_show_false_returns_axes(self, trained_pca):
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        ax = plot_merit(trained_pca, show=False)
        assert isinstance(ax, plt.Axes)


# ======================================================================================
# Undefined lifecycle path tests
# ======================================================================================


class TestSetupAxesDirect:
    """Direct unit tests for ``_setup_axes`` helper."""

    def test_ax_none_creates_subplot(self):
        from spectrochempy.utils.mplutils import _setup_axes

        ax = _setup_axes(None, clear=True)
        assert isinstance(ax, plt.Axes)

    def test_ax_provided_cleared_by_default(self):
        from spectrochempy.utils.mplutils import _setup_axes

        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        assert len(ax.lines) == 1

        ax = _setup_axes(ax)
        assert len(ax.lines) == 0

    def test_ax_provided_clear_false(self):
        from spectrochempy.utils.mplutils import _setup_axes

        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        assert len(ax.lines) == 1

        ax = _setup_axes(ax, clear=False)
        assert len(ax.lines) == 1

    def test_projection_3d(self):
        from mpl_toolkits.mplot3d import Axes3D

        from spectrochempy.utils.mplutils import _setup_axes

        ax = _setup_axes(None, projection="3d")
        assert isinstance(ax, Axes3D)

    def test_ax_none_clear_ignored(self):
        from spectrochempy.utils.mplutils import _setup_axes

        ax = _setup_axes(None, clear=False)
        assert isinstance(ax, plt.Axes)
