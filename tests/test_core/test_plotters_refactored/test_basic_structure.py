# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Minimal test to verify stateless plotting test structure.

This test verifies the basic test infrastructure without relying
on full spectrochempy functionality to isolate testing issues.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest


def assert_dataset_state_unchanged(dataset_before, dataset_after):
    """Mock version for basic test structure verification."""
    # For now, just verify that both datasets have same basic structure
    assert type(dataset_before) == type(dataset_after)
    assert hasattr(dataset_before, '__dict__') == hasattr(dataset_after, '__dict__')
    
    # Verify no plotting attributes added
    assert not hasattr(dataset_after, 'fig')
    assert not hasattr(dataset_after, 'ndaxes')


def get_rcparams_snapshot():
    """Mock rcparams snapshot."""
    import matplotlib as mpl
    return dict(mpl.rcParams)


class MockDataset:
    """Mock dataset class for testing structure."""
    
    def __init__(self):
        self.data = np.array([1, 2, 3, 4, 5])
        self.__dict__ = {'data': self.data}


class TestBasicTestStructure:
    """Test basic test infrastructure."""

    def test_basic_pytest_functionality(self):
        """Verify basic pytest and matplotlib functionality."""
        # Create mock dataset
        dataset = MockDataset()
        ds_before = dataset.__dict__.copy()
        
        # Test basic matplotlib plot creation
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4, 5])
        ax.set_title("Test Plot")
        
        # Verify basic structure checks work
        assert_dataset_state_unchanged(ds_before, dataset)
        
        # Verify rcparams function works
        rcparams = get_rcparams_snapshot()
        assert isinstance(rcparams, dict)
        assert 'axes.titlesize' in rcparams
        
        # Cleanup
        plt.close('all')
        
        # Verify cleanup worked
        assert len(plt.get_fignums()) == 0

    def test_matplotlib_backend_is_agg(self):
        """Verify matplotlib is using Agg backend."""
        backend = plt.get_backend().lower()
        assert 'agg' in backend, f"Expected Agg backend, got {backend}"