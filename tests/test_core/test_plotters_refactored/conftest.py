# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Conftest for stateless plotting tests.

Minimal conftest that forces Agg backend and provides basic fixtures
without depending on full spectrochempy functionality.
"""

import matplotlib
import numpy as np
import pytest

# Force non-interactive backend for all tests
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Auto-cleanup fixture to ensure test independence."""
    yield
    plt.close("all")


def get_rcparams_snapshot():
    """Get current matplotlib rcParams as a dictionary for comparison."""
    import matplotlib as mpl
    return dict(mpl.rcParams)