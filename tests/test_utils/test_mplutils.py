from unittest.mock import MagicMock
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from spectrochempy.utils.meta import Meta
from spectrochempy.utils.mplutils import figure
from spectrochempy.utils.mplutils import get_figure
from spectrochempy.utils.mplutils import get_plotly_figure
from spectrochempy.utils.mplutils import make_label
from spectrochempy.utils.mplutils import show


class TestMplutils:
    def test_figure_creation(self):
        """Test if figure is correctly created with preferences."""
        preferences = Meta()
        preferences.figure_figsize = (8, 6)
        preferences.figure_dpi = 100
        preferences.figure_facecolor = "white"
        preferences.figure_edgecolor = "white"
        preferences.figure_frameon = True
        preferences.figure_autolayout = True

        fig = figure(preferences=preferences)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6
        assert fig.get_dpi() == 100
        plt.close(fig)

    @patch("spectrochempy.NO_DISPLAY", False)  # Ensure NO_DISPLAY is False
    @patch("spectrochempy.utils.mplutils.get_figure")
    @patch("matplotlib.pyplot.show")
    def test_show(self, mock_show, mock_get_figure):
        """Test the show function behavior."""
        # Setup the mock to return a truthy value
        mock_fig = MagicMock()
        mock_get_figure.return_value = (
            mock_fig  # This needs to be truthy for the condition
        )

        # Test show function
        show()
        mock_get_figure.assert_called_once_with(clear=False)
        mock_show.assert_called_once_with(block=True)

    @patch("spectrochempy.NO_DISPLAY", True)
    @patch("spectrochempy.utils.mplutils.plt.close")
    def test_show_no_display(self, mock_close):
        """Test the show function when NO_DISPLAY is True."""
        show()
        mock_close.assert_called_once_with("all")

    def test_get_figure(self):
        """Test get_figure function."""
        preferences = Meta()
        preferences.figure_figsize = (10, 8)
        preferences.figure_dpi = 120
        preferences.figure_facecolor = "white"
        preferences.figure_edgecolor = "white"
        preferences.figure_frameon = True
        preferences.figure_autolayout = False

        fig = get_figure(preferences=preferences, clear=True)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        assert fig.get_dpi() == 120
        plt.close(fig)

    def test_make_label(self):
        """Test make_label function."""

        class MockUnit:
            def __init__(self, unit_str):
                self.unit_str = unit_str

            def __str__(self):
                return self.unit_str

            def __format__(self, format_spec):
                # Handle the ~L format specifier
                if "~L" in format_spec:
                    return self.unit_str
                return self.unit_str

        class MockSpectrum:
            def __init__(self, title, units):
                self.title = title
                self.units = MockUnit(units) if units else None

        # Mock pint.__version__ to avoid version issues
        with patch("pint.__version__", "0.24"):
            # Test with title and units
            spec = MockSpectrum("Absorbance", "cm^-1")
            label = make_label(spec, use_mpl=True)
            assert "Absorbance" in label
            assert "cm^-1" in label

            # Test with no title
            spec = MockSpectrum(None, "cm^-1")
            label = make_label(spec, lab="Intensity", use_mpl=True)
            assert "Intensity" in label
            assert "cm^-1" in label

            # Test with no units
            spec = MockSpectrum("Absorbance", None)
            label = make_label(spec, use_mpl=True)
            assert "Absorbance" in label

            # Test with "<untitled>" in title
            spec = MockSpectrum("<untitled>", "cm^-1")
            label = make_label(spec, use_mpl=True)
            assert "values" in label

            # Test with dimensionless units
            spec = MockSpectrum("Ratio", "dimensionless")
            label = make_label(spec, use_mpl=True)
            assert "Ratio" in label
            assert "dimensionless" not in label

    @pytest.mark.parametrize("plotly_available", [True, False])
    def test_get_plotly_figure(self, plotly_available):
        """Test get_plotly_figure function."""
        if not plotly_available:
            with (
                patch(
                    "spectrochempy.utils.optional.import_optional_dependency",
                    side_effect=ImportError("No module named 'plotly'"),
                ),
                pytest.raises(ImportError),
            ):
                get_plotly_figure()
        else:
            # Mock plotly if available
            mock_go = MagicMock()
            mock_figure = MagicMock()
            mock_go.Figure.return_value = mock_figure

            with patch(
                "spectrochempy.utils.optional.import_optional_dependency",
                return_value=mock_go,
            ):
                # Test creating new figure
                fig = get_plotly_figure(clear=True)
                assert fig == mock_figure
                mock_go.Figure.assert_called_once()

                # Test with existing figure
                existing_fig = MagicMock()
                fig = get_plotly_figure(clear=False, fig=existing_fig)
                assert fig == existing_fig
