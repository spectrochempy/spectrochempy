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

# Skip doctest collection for this file
__doctest_skip__ = ["*"]

import matplotlib
import numpy as np
import pytest

# Force non-interactive backend for all tests
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from spectrochempy import Coord
from spectrochempy import NDDataset


@pytest.fixture
def sample_1d_dataset():
    """Create a deterministic 1D dataset with coordinates and units."""
    np.random.seed(42)  # Deterministic data
    x_data = np.linspace(1, 10, 50)
    y = np.sin(x_data) + np.random.random(50) * 0.1

    x = Coord(data=x_data, title="Wavenumber", units="cm^-1")
    return NDDataset(y, title="Intensity", units="a.u.", coordset=[x])


@pytest.fixture
def sample_2d_dataset():
    """Create a deterministic 2D dataset with coordinates and units."""
    np.random.seed(42)  # Deterministic data
    x_data = np.linspace(1, 5, 20)
    z_data = np.linspace(100, 200, 30)
    data = np.random.random((20, 30)) + np.sin(x_data[:, np.newaxis]) * 0.5

    x = Coord(data=x_data, title="Time", units="s")
    z = Coord(data=z_data, title="Frequency", units="Hz")
    return NDDataset(data, title="Absorbance", units="dimensionless", coordset=[x, z])


@pytest.fixture
def sample_3d_dataset():
    """
    Create a deterministic 2D dataset for 3D visualization (surface, waterfall).

    Note: Despite the name, this creates 2D data for 3D visualization.
    Surface and waterfall plots require 2D height fields, not 3D volumes.
    """
    np.random.seed(42)  # Deterministic data
    x_data = np.linspace(0, 2, 20)
    y_data = np.linspace(0, 1, 15)
    data = np.random.random((15, 20)) + np.sin(x_data[np.newaxis, :]) * 0.5

    x = Coord(data=x_data, title="X", units="nm")
    y = Coord(data=y_data, title="Y", units="µm")
    return NDDataset(data, title="Intensity", units="kJ/mol", coordset=[y, x])


@pytest.fixture
def clean_figures():
    """Explicit fixture for tests that need figure cleanup (backward compatibility)."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def clean_figures_and_plotting_prefs():
    """Auto-cleanup fixture to ensure test independence for plotting tests."""
    from spectrochempy.application.preferences import preferences

    # Reset to defaults BEFORE each plotting test to ensure clean state
    # This prevents pollution from non-plotting tests that ran earlier
    preferences.reset()

    # Snapshot the now-clean state
    snapshot = {
        "colormap": preferences.colormap,
        "colorbar": preferences.colorbar,
        "style": preferences.style,
        "colormap_sequential": preferences.colormap_sequential,
        "colormap_diverging": preferences.colormap_diverging,
        "colormap_categorical_small": preferences.colormap_categorical_small,
        "colormap_categorical_large": preferences.colormap_categorical_large,
        "figure.figsize": preferences.figure.figsize,
        "font.family": preferences.font.family,
        "font.size": preferences.font.size,
        "axes.grid": preferences.axes.grid,
    }

    yield

    # Restore preferences after test (in case test modified them)
    try:
        preferences.reset()
        for key, value in snapshot.items():
            if "." in key:
                group, attr = key.split(".", 1)
                setattr(getattr(preferences, group), attr, value)
            else:
                setattr(preferences, key, value)
    except Exception:
        # If restore fails, at least reset to defaults
        preferences.reset()

    # Close all figures
    plt.close("all")


@pytest.fixture
def backend_checker():
    """Fixture providing backend capability information."""

    class BackendChecker:
        def __init__(self):
            self.backend = matplotlib.get_backend()
            self.is_interactive = plt.isinteractive()
            self.supports_3d = True

    return BackendChecker()


def assert_dataset_state_unchanged(dataset_before, dataset_after):
    """
    Verify that dataset object has no new plotting-related attributes after plotting.

    This is critical for stateless architecture - datasets must remain pure data containers.

    Parameters
    ----------
    dataset_before : dict or object
        Either a dict copy of __dict__ or the dataset object before plotting.
    dataset_after : object
        The dataset object after plotting.
    """
    # Handle both dict and object for dataset_before
    if isinstance(dataset_before, dict):
        before_dict = dataset_before
    else:
        before_dict = dataset_before.__dict__

    # Internal attributes that may be lazily created (not plotting-related)
    internal_attrs = {
        "_NDDataset__mask_metadata",
        "__mask_metadata",
        "_mask_metadata",
    }

    # Check for plotting-related attributes only
    after_dict = dataset_after.__dict__

    # Find new keys that aren't internal lazy-init attributes
    new_keys = set(after_dict.keys()) - set(before_dict.keys())
    plotting_keys = new_keys - internal_attrs

    assert (
        not plotting_keys
    ), f"Dataset object was mutated by plotting with new attributes: {plotting_keys}"

    # No plotting attributes should exist
    assert not hasattr(
        dataset_after, "fig"
    ), "Dataset should not have 'fig' attribute after plotting"
    assert not hasattr(
        dataset_after, "ndaxes"
    ), "Dataset should not have 'ndaxes' attribute after plotting"


def get_rcparams_snapshot():
    """Get current matplotlib rcParams as a dictionary for comparison."""
    import matplotlib as mpl

    return dict(mpl.rcParams)
