# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Tests to verify that docstrings in spectrochempy.plotting are plain text (no docrep).

This test module ensures:
1. No template placeholders (e.g., %(plot.parameters)s) remain in docstrings
2. No SyntaxWarning from docrep are emitted when importing plot modules
"""


class TestDocstringsPlaintext:
    """Test that docstrings contain no template placeholders."""

    def test_no_template_placeholders_in_plot1d(self):
        """Test that plot1d functions have no template placeholders."""
        from spectrochempy.plotting.plot1d import plot_1D
        from spectrochempy.plotting.plot1d import plot_pen
        from spectrochempy.plotting.plot1d import plot_scatter

        for func in [plot_1D, plot_pen, plot_scatter]:
            doc = func.__doc__
            assert doc is not None
            assert "%(" not in doc, f"Found template placeholder in {func.__name__}"

    def test_no_template_placeholders_in_plot2d(self):
        """Test that plot2d functions have no template placeholders."""
        from spectrochempy.plotting.plot2d import plot_2D
        from spectrochempy.plotting.plot2d import plot_image
        from spectrochempy.plotting.plot2d import plot_map
        from spectrochempy.plotting.plot2d import plot_stack

        for func in [plot_2D, plot_map, plot_stack, plot_image]:
            doc = func.__doc__
            assert doc is not None
            assert "%(" not in doc, f"Found template placeholder in {func.__name__}"

    def test_no_template_placeholders_in_plot3d(self):
        """Test that plot3d functions have no template placeholders."""
        from spectrochempy.plotting.plot3d import plot_3D
        from spectrochempy.plotting.plot3d import plot_surface
        from spectrochempy.plotting.plot3d import plot_waterfall

        for func in [plot_3D, plot_surface, plot_waterfall]:
            doc = func.__doc__
            assert doc is not None
            assert "%(" not in doc, f"Found template placeholder in {func.__name__}"


class TestNoSyntaxWarnings:
    """Test that importing plot modules produces no SyntaxWarning."""

    def test_no_syntax_warning_on_import_plot1d(self):
        """Test that importing plot1d doesn't emit SyntaxWarning."""
        # Use subprocess to avoid polluting sys.modules in main process
        import os
        import subprocess
        import sys

        code = """
import warnings
import sys

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from spectrochempy.plotting import plot1d

    syntax_warnings = [x for x in w if issubclass(x.category, SyntaxWarning)]
    if syntax_warnings:
        print(f"FAIL: Found SyntaxWarning: {syntax_warnings}")
        sys.exit(1)
    print("OK: No SyntaxWarning")
    sys.exit(0)
"""

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"SyntaxWarning detected in subprocess!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_no_syntax_warning_on_import_plot2d(self):
        """Test that importing plot2d doesn't emit SyntaxWarning."""
        # Use subprocess to avoid polluting sys.modules in main process
        import os
        import subprocess
        import sys

        code = """
import warnings
import sys

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from spectrochempy.plotting import plot2d

    syntax_warnings = [x for x in w if issubclass(x.category, SyntaxWarning)]
    if syntax_warnings:
        print(f"FAIL: Found SyntaxWarning: {syntax_warnings}")
        sys.exit(1)
    print("OK: No SyntaxWarning")
    sys.exit(0)
"""

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"SyntaxWarning detected in subprocess!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_no_syntax_warning_on_import_plot3d(self):
        """Test that importing plot3d doesn't emit SyntaxWarning."""
        # Use subprocess to avoid polluting sys.modules in main process
        import os
        import subprocess
        import sys

        code = """
import warnings
import sys

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from spectrochempy.plotting import plot3d

    syntax_warnings = [x for x in w if issubclass(x.category, SyntaxWarning)]
    if syntax_warnings:
        print(f"FAIL: Found SyntaxWarning: {syntax_warnings}")
        sys.exit(1)
    print("OK: No SyntaxWarning")
    sys.exit(0)
"""

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"SyntaxWarning detected in subprocess!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_no_syntax_warning_on_import_spectrochempy_plotting(self):
        """Test that importing spectrochempy.plotting doesn't emit SyntaxWarning."""
        # Use subprocess to avoid polluting sys.modules in main process
        import os
        import subprocess
        import sys

        code = """
import warnings
import sys

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from spectrochempy.plotting import plot1d, plot2d, plot3d

    syntax_warnings = [x for x in w if issubclass(x.category, SyntaxWarning)]
    if syntax_warnings:
        print(f"FAIL: Found SyntaxWarning: {syntax_warnings}")
        sys.exit(1)
    print("OK: No SyntaxWarning")
    sys.exit(0)
"""

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"SyntaxWarning detected in subprocess!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
