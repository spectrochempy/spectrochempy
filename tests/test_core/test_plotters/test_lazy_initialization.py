# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Tests for lazy matplotlib initialization system.

These tests verify the public contract:
- matplotlib NOT loaded on import spectrochempy
- matplotlib NOT loaded on NDDataset creation
- matplotlib IS loaded on ds.plot()
- ds.plot() returns Axes object
- Removed internal module raises ImportError
"""

import pytest
import sys
import time
import threading
import os

# Ensure non-interactive backend for tests
os.environ.setdefault("MPLBACKEND", "Agg")


class TestLazyInitializationPerformance:
    """Test performance benefits of lazy initialization."""

    def test_import_performance_without_matplotlib(self):
        """Test that import is fast and matplotlib not loaded."""
        # Clear matplotlib from modules if present
        modules_to_remove = [
            m for m in sys.modules.keys() if m.startswith("matplotlib")
        ]
        for module in modules_to_remove:
            del sys.modules[module]

        # Import should be fast and not load matplotlib
        start_time = time.time()
        import spectrochempy as scp

        import_time = time.time() - start_time

        matplotlib_not_loaded = "matplotlib" not in sys.modules

        assert import_time < 0.5
        assert matplotlib_not_loaded is True

    def test_matplotlib_not_loaded_on_dataset_creation(self):
        """Test that NDDataset creation does not load matplotlib."""
        # Clear matplotlib
        modules_to_remove = [
            m for m in sys.modules.keys() if m.startswith("matplotlib")
        ]
        for module in modules_to_remove:
            del sys.modules[module]

        from spectrochempy import NDDataset

        # Create dataset - should NOT load matplotlib
        ds = NDDataset([1, 2, 3])
        assert "matplotlib" not in sys.modules

    def test_matplotlib_loaded_on_plot(self):
        """Test that matplotlib IS loaded when plotting."""
        from spectrochempy import NDDataset

        # Create dataset
        ds = NDDataset([1, 2, 3, 4, 5])

        # Plot - should load matplotlib
        ax = ds.plot(show=False)

        # Verify matplotlib loaded
        assert "matplotlib" in sys.modules

        # Verify axes returned
        assert ax is not None

    def test_plot_returns_axes_object(self):
        """Test that ds.plot() returns matplotlib Axes."""
        from spectrochempy import NDDataset

        ds = NDDataset([1, 2, 3])
        ax = ds.plot(show=False)

        # Should be a matplotlib Axes object
        assert ax is not None
        assert hasattr(ax, "figure")


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_plotting_code_works(self):
        """Test that existing plotting code continues to work."""
        from spectrochempy import NDDataset

        data = NDDataset.random((5, 5))

        ax = data.plot(show=False)
        assert ax is not None

    def test_functional_api_works(self):
        """Test that functional plotting API works."""
        from spectrochempy import NDDataset
        from spectrochempy.plotting.plot1d import plot_pen

        ds = NDDataset([1, 2, 3, 4, 5])
        ax = plot_pen(ds, show=False)

        assert ax is not None

    def test_preference_setting_unchanged(self):
        """Test that preference setting interface is unchanged."""
        from spectrochempy.application.application import app

        app.plot_preferences.figure_figsize = [8, 6]
        app.plot_preferences.font_size = 12
        app.plot_preferences.style = "scpy"

        assert app.plot_preferences.figure_figsize == (8, 6)
        assert app.plot_preferences.font_size == 12
        assert app.plot_preferences.style == "scpy"

    def test_restore_rcparams_functionality(self):
        """Test that restore_rcparams functionality is preserved."""
        import spectrochempy as scp

        assert hasattr(scp, "restore_rcparams")
        scp.restore_rcparams()


class TestRemovedModuleRaisesError:
    """Test that removed internal modules raise appropriate errors."""

    def test_ndplot_module_removed(self):
        """Test that importing removed ndplot module raises ImportError."""
        with pytest.raises(ImportError):
            from spectrochempy.core.dataset.arraymixins.ndplot import NDPlot


class TestDeprecatedAttributes:
    """Test deprecated attributes raise helpful errors."""

    def test_fig_raises_attribute_error(self):
        """Test that accessing ds.fig raises AttributeError."""
        from spectrochempy import NDDataset

        ds = NDDataset([1, 2, 3])

        with pytest.raises(AttributeError) as excinfo:
            _ = ds.fig

        assert "no longer stored" in str(
            excinfo.value
        ) or "Use the returned axes" in str(excinfo.value)

    def test_ndaxes_raises_attribute_error(self):
        """Test that accessing ds.ndaxes raises AttributeError."""
        from spectrochempy import NDDataset

        ds = NDDataset([1, 2, 3])

        with pytest.raises(AttributeError) as excinfo:
            _ = ds.ndaxes

        assert "no longer stored" in str(
            excinfo.value
        ) or "Use the returned axes" in str(excinfo.value)

    def test_ax_raises_attribute_error(self):
        """Test that accessing ds.ax raises AttributeError."""
        from spectrochempy import NDDataset

        ds = NDDataset([1, 2, 3])

        with pytest.raises(AttributeError) as excinfo:
            _ = ds.ax

        assert "no longer stored" in str(
            excinfo.value
        ) or "Use the returned axes" in str(excinfo.value)


class TestConcurrentPlotting:
    """Test concurrent plot operations."""

    def test_concurrent_plot_calls(self):
        """Test concurrent plot calls work correctly."""
        from spectrochempy import NDDataset

        data = NDDataset.random((5, 5))

        results = []
        errors = []

        def plot_data():
            try:
                ax = data.plot(show=False)
                results.append(ax)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=plot_data) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3
