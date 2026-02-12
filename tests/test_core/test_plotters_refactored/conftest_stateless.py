# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Pytest fixtures for stateless plotting tests.

Ensures deterministic test environment with forced Agg backend
and synthetic datasets for consistent testing.
"""

import matplotlib
import numpy as np
import pytest

# Force non-interactive backend for all tests
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from spectrochempy import NDDataset


@pytest.fixture
def sample_1d_dataset():
    """Create a deterministic 1D dataset with coordinates and units."""
    np.random.seed(42)  # Deterministic data
    x = np.linspace(1, 10, 50)
    y = np.sin(x) + np.random.random(50) * 0.1
    
    dataset = NDDataset(y, title="Intensity", units="a.u.")
    dataset.set_coordset(x, title="Wavenumber", units="cm^-1")
    return dataset


@pytest.fixture
def sample_2d_dataset():
    """Create a deterministic 2D dataset with coordinates and units."""
    np.random.seed(42)  # Deterministic data
    x = np.linspace(1, 5, 20)
    z = np.linspace(100, 200, 30)
    data = np.random.random((20, 30)) + np.sin(x[:, np.newaxis]) * 0.5
    
    dataset = NDDataset(data, title="Absorbance", units="dimensionless")
    dataset.set_coordset(x, title="Time", units="s")
    dataset.set_coordset(z, dim=1, title="Frequency", units="Hz")
    return dataset


@pytest.fixture
def sample_3d_dataset():
    """Create a deterministic 3D dataset with coordinates and units."""
    np.random.seed(42)  # Deterministic data
    x = np.linspace(0, 2, 10)
    y = np.linspace(0, 1, 15) 
    z = np.linspace(0, 1, 20)
    data = np.random.random((10, 15, 20)) + 0.1
    
    dataset = NDDataset(data, title="Intensity", units="kJ/mol")
    dataset.set_coordset(x, title="X", units="nm")
    dataset.set_coordset(y, dim=1, title="Y", units="µm")
    dataset.set_coordset(z, dim=2, title="Z", units="ps")
    return dataset


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Auto-cleanup fixture to ensure test independence."""
    yield
    plt.close("all")


def assert_dataset_state_unchanged(dataset_before, dataset_after):
    """
    Verify that dataset object has no new plotting-related attributes after plotting.
    
    This is critical for stateless architecture - datasets must remain pure data containers.
    """
    # Dataset dictionary must be identical
    assert dataset_before.__dict__ == dataset_after.__dict__, (
        "Dataset object was mutated by plotting - violates stateless architecture"
    )
    
    # No plotting attributes should exist
    assert not hasattr(dataset_after, 'fig'), (
        "Dataset should not have 'fig' attribute after plotting"
    )
    assert not hasattr(dataset_after, 'ndaxes'), (
        "Dataset should not have 'ndaxes' attribute after plotting"
    )


def get_rcparams_snapshot():
    """Get current matplotlib rcParams as a dictionary for comparison."""
    import matplotlib as mpl
    return dict(mpl.rcParams)