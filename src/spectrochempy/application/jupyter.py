# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Jupyter notebook utilities for the SpectroChemPy package."""


def setup_jupyter_css():
    """Set up SpectroChemPy custom CSS for Jupyter notebooks."""
    from pathlib import Path

    from IPython.display import HTML
    from IPython.display import display

    try:
        # Get the custom.css location
        css_file = Path(__file__).parent.parent / "data" / "css" / "custom.css"
        if not css_file.exists():
            return

        # Load and display the CSS
        with open(css_file) as f:
            css = f.read()

        display(HTML(f"<style>{css}</style>"))

    except ImportError:
        # Jupyter not installed
        pass
