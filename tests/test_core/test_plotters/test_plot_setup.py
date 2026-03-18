# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Tests for lazy matplotlib initialization in plot_setup module.

Tests verify that:
- matplotlib import failures raise ImportError
- asset installation failures raise RuntimeError
- _MPL_READY remains False after failures
- successful initialization sets _MPL_READY to True
"""

import builtins
import contextlib
from unittest import mock

import pytest


class TestLazyMplInitialization:
    """Test suite for lazy_ensure_mpl_config initialization logic."""

    def test_successful_initialization_sets_mpl_ready(self):
        """Test that successful initialization sets _MPL_READY to True."""
        import spectrochempy.core.plotters.plot_setup as plot_setup

        plot_setup._MPL_READY = False
        plot_setup._ASSETS_INSTALLED = False

        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        lazy_ensure_mpl_config()

        assert plot_setup._MPL_READY is True
        assert plot_setup._ASSETS_INSTALLED is True

    def test_matplotlib_import_failure_propagates(self):
        """Test that matplotlib import failure propagates as ImportError."""
        import spectrochempy.core.plotters.plot_setup as plot_setup
        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        plot_setup._MPL_READY = False
        plot_setup._ASSETS_INSTALLED = False

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("Simulated matplotlib not found")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="matplotlib is required for plotting"),
        ):
            lazy_ensure_mpl_config()

        assert plot_setup._MPL_READY is False

    def test_asset_installation_failure_raises_runtime_error(self):
        """Test that asset installation failure raises RuntimeError."""
        import spectrochempy.core.plotters.plot_setup as plot_setup
        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        plot_setup._MPL_READY = False
        plot_setup._ASSETS_INSTALLED = False

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                return original_import(name, *args, **kwargs)
            if name == "spectrochempy.core.plotters._mpl_assets":
                raise RuntimeError("Simulated asset installation failure")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(RuntimeError, match="Failed to initialize matplotlib assets"),
        ):
            lazy_ensure_mpl_config()

        assert plot_setup._MPL_READY is False
        assert plot_setup._ASSETS_INSTALLED is False

    def test_mpl_ready_remains_false_after_failure(self):
        """Test that _MPL_READY remains False after initialization failure."""
        import spectrochempy.core.plotters.plot_setup as plot_setup

        original_ready = plot_setup._MPL_READY
        original_installed = plot_setup._ASSETS_INSTALLED

        try:
            plot_setup._MPL_READY = False
            plot_setup._ASSETS_INSTALLED = False

            from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "matplotlib":
                    raise ImportError("Test failure")
                return original_import(name, *args, **kwargs)

            with (
                mock.patch.object(builtins, "__import__", side_effect=mock_import),
                contextlib.suppress(ImportError),
            ):
                lazy_ensure_mpl_config()

            assert (
                plot_setup._MPL_READY is False
            ), "_MPL_READY should be False after failed init"

        finally:
            plot_setup._MPL_READY = original_ready
            plot_setup._ASSETS_INSTALLED = original_installed

    def test_fast_path_works_after_successful_init(self):
        """Test that fast path (early return) works after successful initialization."""
        import spectrochempy.core.plotters.plot_setup as plot_setup

        original_ready = plot_setup._MPL_READY
        original_installed = plot_setup._ASSETS_INSTALLED

        try:
            plot_setup._MPL_READY = True
            plot_setup._ASSETS_INSTALLED = True

            from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

            lazy_ensure_mpl_config()

        finally:
            plot_setup._MPL_READY = original_ready
            plot_setup._ASSETS_INSTALLED = original_installed

    def test_idempotent_behavior(self):
        """Test that lazy_ensure_mpl_config is idempotent."""
        import spectrochempy.core.plotters.plot_setup as plot_setup
        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        original_ready = plot_setup._MPL_READY

        try:
            plot_setup._MPL_READY = False
            plot_setup._ASSETS_INSTALLED = False
            lazy_ensure_mpl_config()
            assert plot_setup._MPL_READY is True

            lazy_ensure_mpl_config()
            assert plot_setup._MPL_READY is True

        finally:
            plot_setup._MPL_READY = original_ready
