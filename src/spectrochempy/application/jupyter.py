from pathlib import Path

from IPython import get_ipython
from IPython.display import HTML
from IPython.display import display


def setup_jupyter_css():
    """Set up SpectroChemPy custom CSS for Jupyter notebooks."""
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


def is_notebook():
    """Check if we are running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
            return True
        if shell == "TerminalInteractiveShell":  # Terminal running IPython
            return False
        return False
    except NameError:  # Probably standard Python interpreter
        return False
