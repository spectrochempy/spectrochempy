# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Startup functions for the spectrochempy package."""


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
