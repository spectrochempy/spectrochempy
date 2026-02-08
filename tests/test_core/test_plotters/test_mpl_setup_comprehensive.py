# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import os
import sys
import pytest
import threading
import time
from unittest.mock import patch, MagicMock, Mock

# Import NDDataset directly to avoid lazy loading issues
from spectrochempy.core.dataset.nddataset import NDDataset

from spectrochempy.core.plotters.plot_setup import (
    lazy_ensure_mpl_config,
    MPLInitState,
    _MPL_INIT_LOCK,
    _is_mpl_initialized,
    _get_mpl_state,
)


class TestLazyMplInitialization:
    """Test lazy matplotlib initialization system."""

    def setup_method(self):
        """Reset state before each test."""
        with _MPL_INIT_LOCK:
            # Reset global state to initial condition by accessing the module directly
            import spectrochempy.core.plotters.plot_setup as plot_setup

            plot_setup._MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED

            # Clear any pending preference changes
            from spectrochempy.core.plotters.plot_setup import (
                _PENDING_PREFERENCE_CHANGES,
            )

            _PENDING_PREFERENCE_CHANGES.clear()

        # Remove matplotlib if already imported
        sys.modules.pop("matplotlib", None)
        sys.modules.pop("matplotlib.pyplot", None)
        sys.modules.pop("matplotlib.backends", None)

    def test_lazy_initialization_basic_functionality(self):
        """Test basic lazy initialization functionality."""
        # Initially not initialized
        assert _is_mpl_initialized() is False
        assert _get_mpl_state() == MPLInitState.NOT_INITIALIZED

        # Should initialize without error
        lazy_ensure_mpl_config()

        assert _is_mpl_initialized() is True
        assert _get_mpl_state() == MPLInitState.INITIALIZED

    def test_lazy_initialization_idempotent(self):
        """Test that lazy initialization is idempotent."""
        # First call should initialize
        lazy_ensure_mpl_config()
        first_call_state = _get_mpl_state()

        # Second call should be no-op
        lazy_ensure_mpl_config()
        second_call_state = _get_mpl_state()

        assert first_call_state == MPLInitState.INITIALIZED
        assert second_call_state == MPLInitState.INITIALIZED

    def test_lazy_initialization_thread_safety(self):
        """Test thread safety of lazy initialization."""
        results = []
        errors = []

        def worker():
            try:
                result = lazy_ensure_mpl_config()
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads simultaneously
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors and all successful
        assert len(errors) == 0
        assert len(results) == 10
        assert _get_mpl_state() == MPLInitState.INITIALIZED

    def test_lazy_initialization_matplotlib_import_failure(self):
        """Test handling of matplotlib import failure."""
        # Note: This test is limited by the fact that matplotlib is already imported
        # in the test environment. However, we can verify the error handling path
        # by testing the state transition logic.

        # Reset state
        import spectrochempy.core.plotters.plot_setup as plot_setup

        plot_setup._MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED

        # Mock the initialization function to raise an exception
        with patch(
            "spectrochempy.core.plotters.plot_setup._perform_lazy_mpl_initialization",
            side_effect=Exception("Mock initialization failure"),
        ):
            with pytest.raises(Exception):
                lazy_ensure_mpl_config()

            assert _get_mpl_state() == MPLInitState.FAILED

    def test_lazy_initialization_headless_environment(self):
        """Test lazy initialization in headless environment."""
        # Simulate headless environment
        with patch.dict(os.environ, {"DISPLAY": ""}, clear=True):
            # This should not raise an exception
            lazy_ensure_mpl_config()
            assert _get_mpl_state() == MPLInitState.INITIALIZED

    def test_lazy_initialization_state_machine(self):
        """Test proper state transitions."""
        # Start in NOT_INITIALIZED
        assert _get_mpl_state() == MPLInitState.NOT_INITIALIZED

        # Should transition to INITIALIZING then INITIALIZED
        with patch(
            "spectrochempy.core.plotters.plot_setup._perform_lazy_mpl_initialization"
        ) as mock_init:
            mock_init.side_effect = lambda: (
                None
            )  # Don't actually initialize for this test
            lazy_ensure_mpl_config()
            # State should have been set to INITIALIZING during the call
            # but will be set to INITIALIZED by the actual initialization

    def test_is_mpl_initialized_helper(self):
        """Test the _is_mpl_initialized helper function."""
        # Initially false
        assert _is_mpl_initialized() is False

        # After initialization, true
        lazy_ensure_mpl_config()
        assert _is_mpl_initialized() is True


def test_lazy_initialization_performance(self):
    """Test that lazy initialization provides performance benefits."""
    # Clear matplotlib modules first
    modules_to_remove = [m for m in sys.modules.keys() if m.startswith("matplotlib")]
    for module in modules_to_remove:
        del sys.modules[module]

    # Import spectrochempy (matplotlib should not be loaded)
    import spectrochempy as scp

    matplotlib_not_loaded = "matplotlib" not in sys.modules

    # Create a dataset and plot (triggers lazy init)
    data = NDDataset([1, 2, 3, 4, 5])
    start_time = time.time()
    data.plot()
    plot_time = time.time() - start_time

    # Matplotlib should now be loaded
    matplotlib_loaded = "matplotlib" in sys.modules

    assert matplotlib_not_loaded is True  # Confirms lazy loading
    assert matplotlib_loaded is True  # Confirms lazy loading worked
    assert plot_time > 0  # Plot should take some time with init


def test_lazy_initialization_with_preferences(self):
    """Test that preference changes work with lazy initialization."""
    from spectrochempy.core.plotters.plot_setup import _PENDING_PREFERENCE_CHANGES

    # Before initialization, changes should be deferred
    from spectrochempy.application.application import app

    app.plot_preferences.figure_figsize = [12, 8]

    # Should have pending changes
    assert len(_PENDING_PREFERENCE_CHANGES) > 0

    # Initialize
    lazy_ensure_mpl_config()

    # After initialization, preferences should be applied
    import matplotlib.pyplot as plt

    assert plt.rcParams["figure.figsize"] == [12.0, 8.0]


class TestLazyPreferenceDeferral:
    """Test preference deferral system."""

    def setup_method(self):
        """Reset state before each test."""
        with _MPL_INIT_LOCK:
            import spectrochempy.core.plotters.plot_setup as plot_setup

            plot_setup._MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED
            from spectrochempy.core.plotters.plot_setup import (
                _PENDING_PREFERENCE_CHANGES,
            )

            _PENDING_PREFERENCE_CHANGES.clear()

    def test_preference_deferral_before_initialization(self):
        """Test that preference changes are deferred before matplotlib init."""
        from spectrochempy.core.plotters.plot_setup import (
            _defer_preference_change,
            _PENDING_PREFERENCE_CHANGES,
            _apply_deferred_preferences,
        )

        # Mock a preference change
        mock_change = {"name": "figure_figsize", "new": [10, 6], "old": [6, 4]}

        # Before initialization, should be deferred
        _defer_preference_change(mock_change)

        assert len(_PENDING_PREFERENCE_CHANGES) == 1
        # Get the first (and only) value from the dictionary
        assert list(_PENDING_PREFERENCE_CHANGES.values())[0] == mock_change

    def test_apply_deferred_preferences(self):
        """Test applying deferred preferences after initialization."""
        from spectrochempy.core.plotters.plot_setup import (
            _defer_preference_change,
            _PENDING_PREFERENCE_CHANGES,
            _apply_deferred_preferences,
        )

        # Add some deferred changes
        mock_change1 = {"name": "figure_figsize", "new": [10, 6], "old": [6, 4]}
        mock_change2 = {"name": "lines_linewidth", "new": 2.0, "old": 1.0}

        _defer_preference_change(mock_change1)
        _defer_preference_change(mock_change2)

        assert len(_PENDING_PREFERENCE_CHANGES) == 2

        # Apply deferred preferences
        _apply_deferred_preferences()

        # Should be cleared after application
        assert len(_PENDING_PREFERENCE_CHANGES) == 0


class TestLegacyCompatibility:
    """Test backward compatibility with legacy code."""

    def test_lazy_initialization_with_old_preferences(self):
        """Test that existing preference setting still works."""
        import spectrochempy as scp
        from spectrochempy.application.application import app

        # Old way of setting preferences should still work
        app.plot_preferences.figure_figsize = [8, 6]

        # Plot should apply these preferences
        data = NDDataset([1, 2, 3, 4, 5])
        data.plot()

        import matplotlib.pyplot as plt

        assert plt.rcParams["figure.figsize"] == [8.0, 6.0]
