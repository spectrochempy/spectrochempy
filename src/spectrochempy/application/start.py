# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Startup functions for the spectrochempy package."""


def display_loading_message(n: int) -> None:
    """
    Display a loading message.

    It's a rich text message that is displayed in the Jupyter Notebook environment.

    Parameters
    ----------
    n : int
        Number of dots to display at the end of the message.

    """
    from spectrochempy.utils.system import is_notebook

    message = f"Loading SpectroChemPy API{'.' * n}"
    if is_notebook():
        from IPython.display import clear_output
        from IPython.display import publish_display_data

        clear_output()
        info = (
            f"<div style='color: black; background-color: #f8d7da; padding: 10px; border: 1px solid #f5c6cb; border-radius: 5px;'>"
            f"{message}</div>"
        )
        publish_display_data(data={"text/html": info})
    else:
        print(message)  # noqa: T201


def set_warnings() -> None:
    """Set warnings for the package."""
    import warnings

    import numpy as np

    warnings.filterwarnings(
        action="once",
        module="spectrochempy",
        category=DeprecationWarning,
    )

    warnings.filterwarnings(
        action="error",
        module="spectrochempy",
        category=np.exceptions.VisibleDeprecationWarning,
    )

    # Ignore warnings from third-party packages
    warnings.filterwarnings(action="ignore", module="jupyter")
    warnings.filterwarnings(action="ignore", module="pykwalify")
    warnings.filterwarnings(action="ignore", module="matplotlib")
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # from pint import UnitStrippedWarning

    # warnings.filterwarnings(action="ignore", category=UnitStrippedWarning)
