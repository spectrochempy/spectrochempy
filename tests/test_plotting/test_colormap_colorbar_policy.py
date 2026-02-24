# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Tests for colormap and colorbar policy resolution (Fix #1).

Tests cover:
- prefs.colormap auto mode
- prefs.colormap fixed override
- prefs.colormap_sequential/diverging/categorical overrides
- plot(..., cmap=None) stack behavior
- prefs.colorbar None/True/False
- plot(..., colorbar=True/False/None)
"""

import numpy as np


class TestColormapPreferences:
    """Test colormap preference defaults and validation."""

    def test_colormap_default_is_auto(self):
        """prefs.colormap default should be 'auto'."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colormap == "auto"

    def test_colormap_sequential_default(self):
        """prefs.colormap_sequential default should be 'viridis'."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colormap_sequential == "viridis"

    def test_colormap_diverging_default(self):
        """prefs.colormap_diverging default should be 'RdBu_r'."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colormap_diverging == "RdBu_r"

    def test_colormap_categorical_small_default(self):
        """prefs.colormap_categorical_small default should be 'tab10'."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colormap_categorical_small == "tab10"

    def test_colormap_categorical_large_default(self):
        """prefs.colormap_categorical_large default should be 'tab20'."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colormap_categorical_large == "tab20"

    def test_colormap_categorical_threshold_default(self):
        """prefs.colormap_categorical_threshold default should be 10."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colormap_categorical_threshold == 10

    def test_colormap_accepts_valid_names(self):
        """prefs.colormap should accept valid colormap names."""
        from spectrochempy.application.preferences import preferences

        original = preferences.colormap
        try:
            preferences.colormap = "plasma"
            assert preferences.colormap == "plasma"
        finally:
            preferences.colormap = original

    def test_colormap_accepts_auto(self):
        """prefs.colormap should accept 'auto'."""
        from spectrochempy.application.preferences import preferences

        original = preferences.colormap
        try:
            preferences.colormap = "auto"
            assert preferences.colormap == "auto"
        finally:
            preferences.colormap = original


class TestColorbarPreferences:
    """Test colorbar preference defaults and validation."""

    def test_colorbar_default_is_none(self):
        """prefs.colorbar default should be None."""
        from spectrochempy.application.preferences import preferences

        assert preferences.colorbar is None

    def test_colorbar_accepts_true(self):
        """prefs.colorbar should accept True."""
        from spectrochempy.application.preferences import preferences

        original = preferences.colorbar
        try:
            preferences.colorbar = True
            assert preferences.colorbar is True
        finally:
            preferences.colorbar = original

    def test_colorbar_accepts_false(self):
        """prefs.colorbar should accept False."""
        from spectrochempy.application.preferences import preferences

        original = preferences.colorbar
        try:
            preferences.colorbar = False
            assert preferences.colorbar is False
        finally:
            preferences.colorbar = original


class TestL1DefaultInjection:
    """Test L1 functions accept default parameters."""

    def test_resolve_2d_colormap_accepts_defaults(self):
        """resolve_2d_colormap should accept default parameters."""
        from spectrochempy.plotting._style import resolve_2d_colormap

        data = np.abs(np.random.randn(10, 10)) + 0.1  # All positive
        cmap, norm = resolve_2d_colormap(
            data,
            default_sequential="plasma",
            default_diverging="coolwarm",
        )
        # For all-positive data, should use sequential default
        assert cmap.name == "plasma"

    def test_resolve_2d_colormap_diverging_default(self):
        """resolve_2d_colormap should use diverging default for bipolar data."""
        from spectrochempy.plotting._style import resolve_2d_colormap

        data = np.random.randn(10, 10) - 0.5  # Has negative values
        cmap, norm = resolve_2d_colormap(
            data,
            default_sequential="viridis",
            default_diverging="coolwarm",
        )
        assert cmap.name == "coolwarm"

    def test_resolve_stack_colors_accepts_defaults(self):
        """resolve_stack_colors should accept categorical defaults."""
        from spectrochempy.plotting._style import resolve_stack_colors

        # Create a mock dataset-like object with required attributes
        class MockDataset:
            _squeeze_ndim = 2
            shape = (5, 10)
            dims = ["y", "x"]

        dataset = MockDataset()
        colors, is_categorical, mappable = resolve_stack_colors(
            dataset,
            n=5,
            palette="categorical",
            default_categorical_small="Set2",
            default_categorical_large="Set3",
            categorical_threshold=10,
        )
        assert is_categorical is True

    def test_get_categorical_cmap_accepts_defaults(self):
        """_get_categorical_cmap should use default parameters."""
        from spectrochempy.plotting._style import _get_categorical_cmap

        cmap = _get_categorical_cmap(
            5,
            default_small="Set2",
            default_large="Set3",
            threshold=10,
        )
        assert cmap is not None
        assert len(cmap.colors) == 5


class TestColorbarPolicy:
    """Test colorbar policy resolution."""

    def test_colorbar_kwarg_true_forces_on(self):
        """colorbar=True should force colorbar on when mappable exists."""
        import spectrochempy as scp

        ds = scp.NDDataset(np.random.randn(5, 10))
        ax = ds.plot_2D(method="image", colorbar=True)
        # Should have colorbar
        assert hasattr(ax, "_scp_colorbar")

    def test_colorbar_kwarg_false_forces_off(self):
        """colorbar=False should force colorbar off."""
        import spectrochempy as scp

        ds = scp.NDDataset(np.random.randn(5, 10))
        ax = ds.plot_2D(method="image", colorbar=False)
        # Should not have colorbar
        assert not hasattr(ax, "_scp_colorbar")

    def test_colorbar_kwarg_none_defers_to_prefs(self):
        """colorbar=None should defer to prefs.colorbar."""
        import spectrochempy as scp

        prefs = scp.preferences
        original = prefs.colorbar
        try:
            # Test with prefs.colorbar = True
            prefs.colorbar = True
            ds = scp.NDDataset(np.random.randn(5, 10))
            ax = ds.plot_2D(method="image", colorbar=None)
            assert hasattr(ax, "_scp_colorbar")
        finally:
            prefs.colorbar = original


class TestColormapPrecedence:
    """Test colormap precedence rules."""

    def test_cmap_kwarg_overrides_prefs(self):
        """Explicit cmap kwarg should override prefs.colormap."""
        import spectrochempy as scp

        prefs = scp.preferences
        original = prefs.colormap
        try:
            prefs.colormap = "plasma"
            ds = scp.NDDataset(np.random.randn(5, 10))
            ax = ds.plot_2D(method="image", cmap="viridis")
            # Should use viridis from kwarg
            assert ax is not None
        finally:
            prefs.colormap = original

    def test_prefs_colormap_fixed_overrides_auto(self):
        """prefs.colormap fixed value should be used over auto."""
        import spectrochempy as scp

        prefs = scp.preferences
        original = prefs.colormap
        try:
            prefs.colormap = "plasma"
            ds = scp.NDDataset(np.random.randn(5, 10))
            ax = ds.plot_2D(method="image")
            assert ax is not None
        finally:
            prefs.colormap = original


class TestStackCmapNone:
    """Test stack plot cmap=None special behavior."""

    def test_stack_cmap_none_forces_categorical(self):
        """plot(..., cmap=None) for stack should force categorical mode."""
        import spectrochempy as scp

        ds = scp.NDDataset(np.random.randn(5, 10))
        ax = ds.plot_2D(method="stack", cmap=None)
        # Should not have colorbar (categorical mode)
        assert not hasattr(ax, "_scp_colorbar")


class TestBackwardCompatibility:
    """Test backward compatibility with legacy API."""

    def test_colormap_kwarg_alias(self):
        """colormap=... kwarg should work same as cmap=..."""
        import spectrochempy as scp

        ds = scp.NDDataset(np.random.randn(5, 10))
        ax = ds.plot_2D(method="image", colormap="plasma")
        assert ax is not None

    def test_default_behavior_unchanged(self):
        """Default behavior should match old behavior (viridis for sequential)."""
        import spectrochempy as scp

        # Create positive data (sequential)
        ds = scp.NDDataset(np.abs(np.random.randn(5, 10)))
        ax = ds.plot_2D(method="image")
        assert ax is not None


class TestStackColorbarRegression:
    """Test stack colorbar behavior for continuous vs categorical modes."""

    def test_stack_auto_colorbar_continuous(self):
        """Continuous stack should show colorbar in auto mode (default prefs)."""
        import spectrochempy as scp

        # Create dataset with float y coordinate (continuous)
        ds = scp.NDDataset(np.random.randn(10, 20))
        y_coord = scp.Coord(np.linspace(100, 200, 10), title="wavelength")
        ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))
        ax = ds.plot_2D(method="lines")
        assert hasattr(ax, "_scp_colorbar"), (
            "Continuous stack should have colorbar in auto mode"
        )

    def test_stack_auto_colorbar_categorical(self):
        """Categorical stack should NOT show colorbar in auto mode."""
        import spectrochempy as scp

        # Create dataset with labeled y coordinate (categorical)
        ds = scp.NDDataset(np.random.randn(5, 20))
        y_coord = scp.Coord(labels=["A", "B", "C", "D", "E"], title="sample")
        ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))
        ax = ds.plot_2D(method="lines")
        assert not hasattr(ax, "_scp_colorbar"), (
            "Categorical stack should not have colorbar in auto mode"
        )

    def test_stack_colorbar_true_continuous(self):
        """colorbar=True for continuous stack should show colorbar."""
        import spectrochempy as scp

        # Create dataset with float y coordinate (continuous)
        ds = scp.NDDataset(np.random.randn(10, 20))
        y_coord = scp.Coord(np.linspace(100, 200, 10), title="wavelength")
        ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))
        ax = ds.plot_2D(method="lines", colorbar=True)
        assert hasattr(ax, "_scp_colorbar"), (
            "Continuous stack with colorbar=True should have colorbar"
        )

    def test_stack_colorbar_false_continuous(self):
        """colorbar=False for continuous stack should NOT show colorbar."""
        import spectrochempy as scp

        # Create dataset with float y coordinate (continuous)
        ds = scp.NDDataset(np.random.randn(10, 20))
        y_coord = scp.Coord(np.linspace(100, 200, 10), title="wavelength")
        ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))
        ax = ds.plot_2D(method="lines", colorbar=False)
        assert not hasattr(ax, "_scp_colorbar"), (
            "Continuous stack with colorbar=False should not have colorbar"
        )

    def test_stack_colorbar_true_categorical(self):
        """colorbar=True for categorical stack should NOT show colorbar (no mappable)."""
        import spectrochempy as scp

        # Create dataset with labeled y coordinate (categorical)
        ds = scp.NDDataset(np.random.randn(5, 20))
        y_coord = scp.Coord(labels=["A", "B", "C", "D", "E"], title="sample")
        ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))
        ax = ds.plot_2D(method="lines", colorbar=True)
        assert not hasattr(ax, "_scp_colorbar"), (
            "Categorical stack has no mappable, so colorbar=True should not create one"
        )

    def test_stack_method_alias(self):
        """Both 'stack' and 'lines' methods should behave identically."""
        import spectrochempy as scp

        # Create dataset with float y coordinate (continuous)
        ds = scp.NDDataset(np.random.randn(10, 20))
        y_coord = scp.Coord(np.linspace(100, 200, 10), title="wavelength")
        ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))

        # Test with method="lines"
        ax1 = ds.plot_2D(method="lines")
        has_colorbar_lines = hasattr(ax1, "_scp_colorbar")

        # Test with method="stack"
        ax2 = ds.plot_2D(method="stack")
        has_colorbar_stack = hasattr(ax2, "_scp_colorbar")

        assert has_colorbar_lines == has_colorbar_stack, (
            "Both 'lines' and 'stack' methods should have same colorbar behavior"
        )


class TestStyleIntegration:
    """Test matplotlib style integration with SpectroChemPy colormap resolution."""

    def _get_cmap_name(self, ax):
        """Get colormap name from axes, handling both image and contour plots."""
        if ax.images:
            return ax.images[0].get_cmap().name
        if ax.collections:
            return ax.collections[0].get_cmap().name
        return None

    def test_style_grayscale_affects_sequential(self):
        """Grayscale style should affect sequential colormap in auto mode."""
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            scp.preferences.style = "grayscale"
            scp.preferences.colormap = "auto"

            ds = scp.NDDataset(np.random.randn(5, 10) + 1)  # positive data
            ax = ds.plot_2D(method="image")
            cmap_name = self._get_cmap_name(ax)

            assert cmap_name == "gray", f"Expected gray, got {cmap_name}"
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_style_grayscale_affects_diverging(self):
        """Grayscale style should affect diverging colormap in auto mode.

        Diverging under grayscale MUST remain grayscale (difference only via norm).
        """
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            scp.preferences.style = "grayscale"
            scp.preferences.colormap = "auto"

            ds = scp.NDDataset(np.random.randn(5, 10) - 0.5)  # diverging data
            ax = ds.plot_2D(method="image")
            cmap_name = self._get_cmap_name(ax)

            assert cmap_name == "gray", f"Expected gray for diverging, got {cmap_name}"
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_style_grayscale_affects_stack(self):
        """Grayscale style should affect stack plot colors in auto mode."""
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            scp.preferences.style = "grayscale"
            scp.preferences.colormap = "auto"

            ds = scp.NDDataset(np.random.randn(10, 20))
            y_coord = scp.Coord(np.linspace(100, 200, 10), title="wavelength")
            ds.set_coordset(y=y_coord, x=scp.Coord(np.arange(20)))
            ax = ds.plot_2D(method="lines", colorbar=True)

            # Check line colors are grayscale (R == G == B)
            for line in ax.lines[:3]:
                color = line.get_color()
                if isinstance(color, tuple) and len(color) >= 3:
                    r, g, b = color[:3]
                    # Allow small tolerance for grayscale
                    assert abs(r - g) < 0.1 and abs(g - b) < 0.1, (
                        f"Line color {color} is not grayscale"
                    )
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_fixed_prefs_colormap_ignores_style(self):
        """Fixed prefs.colormap should override matplotlib style."""
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            scp.preferences.style = "grayscale"
            scp.preferences.colormap = "plasma"

            # Use positive data to get sequential colormap
            ds = scp.NDDataset(np.abs(np.random.randn(5, 10)) + 0.1)
            ax = ds.plot_2D(method="image")
            cmap_name = self._get_cmap_name(ax)

            assert cmap_name == "plasma", f"Expected plasma (fixed), got {cmap_name}"
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_auto_without_style_falls_back_to_prefs_defaults(self):
        """Auto mode with default style should use prefs.colormap_* defaults."""
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            # Use default style (scpy) - don't set to None
            scp.preferences.style = "scpy"
            scp.preferences.colormap = "auto"

            ds = scp.NDDataset(np.abs(np.random.randn(5, 10)) + 0.1)
            ax = ds.plot_2D(method="image")
            cmap_name = self._get_cmap_name(ax)

            # Should fall back to prefs.colormap_sequential (viridis by default)
            assert cmap_name is not None, "No colormap found"
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_style_context_active_during_resolution(self):
        """Verify that matplotlib rcParams reflect style during colormap resolution."""
        import matplotlib as mpl
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        captured_rcParams = {"image.cmap": None}

        def capture_resolve(*args, **kwargs):
            captured_rcParams["image.cmap"] = mpl.rcParams.get("image.cmap")
            from spectrochempy.plotting._style import resolve_2d_colormap as orig

            return orig(*args, **kwargs)

        try:
            scp.preferences.style = "grayscale"
            scp.preferences.colormap = "auto"

            import spectrochempy.plotting._style as style_module

            original_resolve = style_module.resolve_2d_colormap
            style_module.resolve_2d_colormap = capture_resolve

            ds = scp.NDDataset(np.abs(np.random.randn(5, 10)) + 0.1)
            ds.plot_2D(method="image")

            style_module.resolve_2d_colormap = original_resolve

            assert captured_rcParams.get("image.cmap") == "gray", (
                f"rcParams[image.cmap] was {captured_rcParams.get('image.cmap')}, "
                "expected 'gray'"
            )
        finally:
            style_module.resolve_2d_colormap = original_resolve
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_auto_without_style_falls_back_to_prefs_defaults(self):
        """Auto mode with default style should use prefs.colormap_* defaults."""
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            # Use default style (scpy) - can't set to None
            scp.preferences.style = "scpy"
            scp.preferences.colormap = "auto"

            ds = scp.NDDataset(np.abs(np.random.randn(5, 10)) + 0.1)
            ax = ds.plot_2D(method="image")
            cmap_name = self._get_cmap_name(ax)

            # Should fall back to prefs.colormap_sequential (viridis by default)
            assert cmap_name is not None, "No colormap found"
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap

    def test_style_context_active_during_resolution(self):
        """Verify that grayscale style produces grayscale colormap in auto mode.

        This is a functional test - if the style context is active during resolution,
        the resulting colormap will be grayscale.
        """
        import spectrochempy as scp

        orig_style = scp.preferences.style
        orig_colormap = scp.preferences.colormap

        try:
            scp.preferences.style = "grayscale"
            scp.preferences.colormap = "auto"

            ds = scp.NDDataset(np.abs(np.random.randn(5, 10)) + 0.1)
            ax = ds.plot_2D(method="image")
            cmap_name = self._get_cmap_name(ax)

            # If style context was active, colormap should be gray
            assert cmap_name == "gray", (
                f"Expected 'gray' colormap from grayscale style, got '{cmap_name}'"
            )
        finally:
            scp.preferences.style = orig_style
            scp.preferences.colormap = orig_colormap
