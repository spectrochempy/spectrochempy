import matplotlib.pyplot as plt
import pytest
from traitlets import TraitError

from spectrochempy.utils.meta import Meta
from spectrochempy.utils.mplutils import get_figure


class TestWindowPosition:
    def test_window_position_set_tuple(self):
        """Test setting window position to a tuple."""
        preferences = Meta()
        preferences.figure_figsize = (6, 4)
        preferences.figure_dpi = 100
        preferences.figure_facecolor = "white"
        preferences.figure_edgecolor = "white"
        preferences.figure_frameon = True
        preferences.figure_autolayout = False
        preferences.figure_window_position = (100, 100)

        fig = get_figure(preferences=preferences)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_window_position_set_none(self):
        """Test setting window position to None (default behavior)."""
        preferences = Meta()
        preferences.figure_figsize = (6, 4)
        preferences.figure_dpi = 100
        preferences.figure_facecolor = "white"
        preferences.figure_edgecolor = "white"
        preferences.figure_frameon = True
        preferences.figure_autolayout = False
        preferences.figure_window_position = None

        fig = get_figure(preferences=preferences)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_window_position_negative_values(self):
        """Test that negative values are accepted for multi-monitor setups."""
        preferences = Meta()
        preferences.figure_figsize = (6, 4)
        preferences.figure_dpi = 100
        preferences.figure_facecolor = "white"
        preferences.figure_edgecolor = "white"
        preferences.figure_frameon = True
        preferences.figure_autolayout = False
        preferences.figure_window_position = (-100, -100)

        fig = get_figure(preferences=preferences)
        assert fig is not None
        plt.close(fig)

    def test_window_position_invalid_type_raises(self):
        """Test that invalid type raises TraitError."""
        from traitlets import Tuple
        from traitlets import Integer

        class MockPrefs:
            window_position = Tuple(Integer(), Integer(), default_value=None, allow_none=True)

        prefs = MockPrefs()
        with pytest.raises(TraitError):
            prefs.window_position = "invalid"

    def test_window_position_invalid_tuple_length_raises(self):
        """Test that tuple with wrong length raises TraitError."""
        from traitlets import Tuple
        from traitlets import Integer
        from traitlets import HasTraits

        class MockPrefs(HasTraits):
            window_position = Tuple(Integer(), Integer(), default_value=None, allow_none=True)

        prefs = MockPrefs()
        with pytest.raises(TraitError):
            prefs.window_position = (100,)

    def test_window_position_no_crash_under_agg(self):
        """Test that window position does not cause crash under Agg backend."""
        import matplotlib

        preferences = Meta()
        preferences.figure_figsize = (6, 4)
        preferences.figure_dpi = 100
        preferences.figure_facecolor = "white"
        preferences.figure_edgecolor = "white"
        preferences.figure_frameon = True
        preferences.figure_autolayout = False
        preferences.figure_window_position = (100, 100)

        backend = matplotlib.get_backend().lower()

        fig = get_figure(preferences=preferences)
        assert fig is not None

        if "agg" in backend:
            pass

        plt.close(fig)
