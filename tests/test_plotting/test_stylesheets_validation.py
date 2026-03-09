# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Validation tests for SpectroChemPy stylesheet functionality.

Tests ensure that:
1. Stylesheets are present and accessible
2. scpy style modifies rcParams meaningfully
3. scpy neutralizes prior global style
4. paper style changes DPI and figsize
5. sans style changes font.family
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


class TestStylesheetPresence:
    """Test that stylesheets are present in the package."""

    def test_stylesheets_directory_exists(self):
        """Verify the canonical stylesheets directory exists."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        assert (
            stylesheets_path.exists()
        ), f"Stylesheets directory not found: {stylesheets_path}"

    def test_scpy_stylesheet_exists(self):
        """Verify scpy.mplstyle exists."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        scpy_path = Path(stylesheets_path) / "scpy.mplstyle"
        assert scpy_path.exists(), f"scpy.mplstyle not found: {scpy_path}"

    def test_paper_stylesheet_exists(self):
        """Verify paper.mplstyle exists."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        paper_path = Path(stylesheets_path) / "paper.mplstyle"
        assert paper_path.exists(), f"paper.mplstyle not found: {paper_path}"

    def test_sans_stylesheet_exists(self):
        """Verify sans.mplstyle exists."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        sans_path = Path(stylesheets_path) / "sans.mplstyle"
        assert sans_path.exists(), f"sans.mplstyle not found: {sans_path}"


class TestScpyStyleContent:
    """Test that scpy style has meaningful content."""

    def test_scpy_not_empty(self):
        """Verify scpy.mplstyle is not a stub (has multiple lines)."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        scpy_path = Path(stylesheets_path) / "scpy.mplstyle"
        content = scpy_path.read_text()

        lines = [
            line
            for line in content.split("\n")
            if line.strip() and not line.startswith("#")
        ]
        assert (
            len(lines) >= 5
        ), f"scpy.mplstyle should have at least 5 non-comment lines, got {len(lines)}"

    def test_scpy_modifies_rcparams(self):
        """Verify scpy style modifies rcParams meaningfully within context."""
        params_to_check = [
            "axes.titlesize",
            "font.family",
            "lines.linewidth",
            "figure.figsize",
        ]

        modified_count = 0
        with plt.style.context("scpy"):
            for param in params_to_check:
                current = mpl.rcParams[param]
                default = mpl.rcParamsDefault.get(param, None)
                if current != default:
                    modified_count += 1

        assert modified_count >= 2, (
            f"scpy style should modify at least 2 rcParams from defaults. "
            f"Modified {modified_count} of {len(params_to_check)} checked params."
        )

    def test_scpy_neutralizes_dark_background(self):
        """Verify scpy neutralizes a prior global dark_background style."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        scpy_path = Path(stylesheets_path) / "scpy.mplstyle"

        plt.style.use("dark_background")

        with plt.style.context(str(scpy_path)):
            facecolor = mpl.rcParams["figure.facecolor"]
            assert (
                facecolor == "white"
            ), f"scpy should set figure.facecolor to white, even after dark_background. Got: {facecolor}"

        plt.style.use("default")


class TestPaperStyleContent:
    """Test that paper style has meaningful content."""

    def test_paper_not_empty(self):
        """Verify paper.mplstyle is not a stub (has multiple lines)."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        paper_path = Path(stylesheets_path) / "paper.mplstyle"
        content = paper_path.read_text()

        lines = [
            line
            for line in content.split("\n")
            if line.strip() and not line.startswith("#")
        ]
        assert (
            len(lines) >= 3
        ), f"paper.mplstyle should have at least 3 non-comment lines, got {len(lines)}"

    def test_paper_changes_dpi(self):
        """Verify paper style changes DPI from default."""
        with plt.style.context("paper"):
            dpi = mpl.rcParams["figure.dpi"]

        assert dpi >= 150, f"paper style should have DPI >= 150, got {dpi}"

    def test_paper_changes_figsize(self):
        """Verify paper style changes figsize from default."""
        default_figsize = mpl.rcParamsDefault["figure.figsize"]

        with plt.style.context("paper"):
            paper_figsize = mpl.rcParams["figure.figsize"]

        assert (
            paper_figsize != default_figsize
        ), f"paper style should change figsize from default {default_figsize}"


class TestSansStyleContent:
    """Test that sans style has meaningful content."""

    def test_sans_not_empty(self):
        """Verify sans.mplstyle is not a stub."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        sans_path = Path(stylesheets_path) / "sans.mplstyle"
        content = sans_path.read_text()

        lines = [
            line
            for line in content.split("\n")
            if line.strip() and not line.startswith("#")
        ]
        assert (
            len(lines) >= 1
        ), f"sans.mplstyle should have at least 1 non-comment line, got {len(lines)}"

    def test_sans_changes_font_family(self):
        """Verify sans style sets font.family to sans-serif."""
        from spectrochempy.utils.packages import get_pkg_path

        stylesheets_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        sans_path = Path(stylesheets_path) / "sans.mplstyle"

        with plt.style.context(str(sans_path)):
            family = mpl.rcParams["font.family"]

        if isinstance(family, list):
            assert (
                "sans-serif" in family
            ), f"sans style should include 'sans-serif' in font.family, got {family}"
        else:
            assert (
                family == "sans-serif"
            ), f"sans style should set font.family to 'sans-serif', got {family}"


class TestPrefsStylesheetsDefault:
    """Test that prefs.stylesheets points to canonical directory."""

    def test_prefs_stylesheets_default(self):
        """Verify prefs.stylesheets default points to canonical location."""
        from spectrochempy.application.preferences import preferences
        from spectrochempy.utils.packages import get_pkg_path

        expected_path = get_pkg_path("plotting/stylesheets", "spectrochempy")
        actual_path = preferences.stylesheets

        assert actual_path is not None, "prefs.stylesheets should not be None"

        actual = Path(actual_path).resolve()
        expected = Path(expected_path).resolve()

        assert actual == expected, (
            f"prefs.stylesheets should point to canonical location.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )
