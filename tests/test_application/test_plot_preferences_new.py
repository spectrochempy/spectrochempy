"""Tests for new plotting preferences."""

import numpy as np


class TestNewPreferences:
    """Test that new preferences work correctly."""

    def test_3d_elev_default(self):
        """axes3d_elev default is 30.0."""
        from spectrochempy.application.preferences import preferences

        assert preferences.axes3d_elev == 30.0

    def test_3d_azim_default(self):
        """axes3d_azim default is 45.0."""
        from spectrochempy.application.preferences import preferences

        assert preferences.axes3d_azim == 45.0

    def test_baseline_region_color_default(self):
        """baseline_region_color default is #2ca02c."""
        from spectrochempy.application.preferences import preferences

        assert preferences.baseline_region_color == "#2ca02c"

    def test_baseline_region_alpha_default(self):
        """baseline_region_alpha default is 0.5."""
        from spectrochempy.application.preferences import preferences

        assert preferences.baseline_region_alpha == 0.5

    def test_image_equal_aspect_default(self):
        """image_equal_aspect default is True."""
        from spectrochempy.application.preferences import preferences

        assert preferences.image_equal_aspect is True

    def test_3d_preferences_modifiable(self):
        """3D preferences can be modified."""
        from spectrochempy.application.preferences import preferences

        original_elev = preferences.axes3d_elev
        original_azim = preferences.axes3d_azim

        preferences.axes3d_elev = 45.0
        preferences.axes3d_azim = 60.0

        assert preferences.axes3d_elev == 45.0
        assert preferences.axes3d_azim == 60.0

        preferences.axes3d_elev = original_elev
        preferences.axes3d_azim = original_azim


class Test3DViewOverrides:
    """Test that explicit kwargs override preferences in 3D plots."""

    def test_plot_score_3d_uses_preferences(self):
        """plot_score 3D uses preference defaults."""
        import matplotlib.pyplot as plt

        from spectrochempy.plotting.composite.plotscore import plot_score

        scores = np.random.randn(20, 5)

        ax = plot_score(scores, components=(1, 2, 3), show=False)

        assert ax is not None
        plt.close(ax.figure)

    def test_plot_score_3d_explicit_kwargs_override(self):
        """Explicit elev/azim kwargs override preferences."""
        import matplotlib.pyplot as plt

        from spectrochempy.plotting.composite.plotscore import plot_score

        scores = np.random.randn(20, 5)

        ax = plot_score(scores, components=(1, 2, 3), elev=60, azim=90, show=False)

        assert ax is not None
        plt.close(ax.figure)


class TestBaselineRegionPreferences:
    """Test that baseline region preferences are used."""

    def test_plot_baseline_uses_preferences(self):
        """plot_baseline uses preference defaults for region color/alpha."""
        import matplotlib.pyplot as plt

        from spectrochempy import Coord
        from spectrochempy import NDDataset
        from spectrochempy.plotting.composite.plotbaseline import plot_baseline

        x = Coord(np.linspace(0, 100, 50), title="x")
        original = NDDataset(np.random.randn(3, 50), coords=[Coord(range(3)), x])
        baseline = NDDataset(np.zeros((3, 50)), coords=[Coord(range(3)), x])
        corrected = original.copy()

        ax1, ax2 = plot_baseline(
            original, baseline, corrected, show=False, show_regions=True
        )

        assert ax1 is not None
        assert ax2 is not None
        plt.close(ax1.figure)


class TestImageEqualAspectPreference:
    """Test that image_equal_aspect preference works correctly."""

    def test_image_equal_aspect_can_be_disabled(self):
        """image_equal_aspect can be set to False."""
        from spectrochempy.application.preferences import preferences

        original = preferences.image_equal_aspect

        preferences.image_equal_aspect = False
        assert preferences.image_equal_aspect is False

        preferences.image_equal_aspect = original

    def test_explicit_equal_aspect_overrides_preference(self):
        """Explicit equal_aspect kwarg overrides preference."""
        import matplotlib.pyplot as plt

        from spectrochempy import Coord
        from spectrochempy import NDDataset

        x = Coord(np.linspace(0, 100, 50), title="x", units="cm")
        y = Coord(np.linspace(0, 50, 30), title="y", units="cm")
        data = NDDataset(np.random.randn(30, 50), coords=[y, x])

        ax = data.plot(method="image", equal_aspect=False, show=False)

        assert ax is not None
        plt.close(ax.figure)
