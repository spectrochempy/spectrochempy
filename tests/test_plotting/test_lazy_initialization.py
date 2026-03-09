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

import os
import subprocess
import sys
import threading

import pytest

# Ensure non-interactive backend for tests
os.environ.setdefault("MPLBACKEND", "Agg")


def _run_in_subprocess(code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """
    Run code in a fresh Python subprocess to avoid polluting the main process.

    Parameters
    ----------
    code : str
        Python code to execute in subprocess.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process with stdout, stderr, and returncode.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["MPLBACKEND"] = "Agg"

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


class TestLazyInitializationPerformance:
    """Test performance benefits of lazy initialization."""

    def test_import_performance_without_matplotlib(self):
        """Test that import is fast and matplotlib not loaded."""
        code = """
import sys
import time

start_time = time.time()

# Import spectrochempy - this should be fast and not load matplotlib
import spectrochempy

import_time = time.time() - start_time

# Check if matplotlib was loaded
matplotlib_loaded = "matplotlib" in sys.modules

# Print results for assertion
print(f"IMPORT_TIME:{import_time}")
print(f"MATPLOTLIB_LOADED:{matplotlib_loaded}")

if matplotlib_loaded:
    # Find which matplotlib modules were loaded
    mpl_modules = [m for m in sys.modules if m.startswith("matplotlib")]
    print(f"MPL_MODULES:{mpl_modules[:10]}")

sys.exit(0 if not matplotlib_loaded else 1)
"""
        result = _run_in_subprocess(code)

        assert result.returncode == 0, (
            f"matplotlib was loaded during import spectrochempy!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Also verify import was fast (informational)
        for line in result.stdout.split("\n"):
            if line.startswith("IMPORT_TIME:"):
                import_time = float(line.split(":")[1])
                assert import_time < 0.5, f"Import took {import_time}s, expected < 0.5s"
                break

    def test_matplotlib_not_loaded_on_dataset_creation(self):
        """Test that NDDataset creation does not load matplotlib."""
        code = """
import sys

# Import spectrochempy and create NDDataset - should NOT load matplotlib
from spectrochempy import NDDataset

# Create dataset - should NOT load matplotlib
ds = NDDataset([1, 2, 3])

# Check if matplotlib was loaded
matplotlib_loaded = "matplotlib" in sys.modules

print(f"MATPLOTLIB_LOADED:{matplotlib_loaded}")

if matplotlib_loaded:
    mpl_modules = [m for m in sys.modules if m.startswith("matplotlib")]
    print(f"MPL_MODULES:{mpl_modules}")

sys.exit(0 if not matplotlib_loaded else 1)
"""
        result = _run_in_subprocess(code)

        assert result.returncode == 0, (
            f"matplotlib was loaded during NDDataset creation!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

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
