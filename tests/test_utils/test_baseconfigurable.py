# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the BaseConfigurable class."""

import logging

import numpy as np
import pytest
import traitlets as tr

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.exceptions import NotTransformedError


class SimpleModel(BaseConfigurable):
    """A simple model for testing purposes."""

    test_param = tr.Bool(False).tag(config=True)


class TestBaseConfigurable:
    """Tests for the BaseConfigurable class."""

    def test_initialization(self):
        """Test initialization of the BaseConfigurable class."""
        # Test with default parameters
        model = SimpleModel()
        assert not model.test_param
        assert not model._applied
        assert model._output_type == "NDDataset"

        # Test with custom parameters
        model = SimpleModel(test_param=True)
        assert model.test_param
        assert not model._applied

        # Test with invalid parameter
        with pytest.raises(KeyError):
            SimpleModel(invalid_param=True)

    def test_log_level(self):
        """Test setting log level."""
        model = SimpleModel(log_level=logging.DEBUG)  # noqa: F841
        # Check if log level is set correctly
        from spectrochempy.application.application import app

        assert app.log.level == logging.DEBUG

    def test_make_dataset(self):
        """Test _make_dataset method."""
        model = SimpleModel()

        # Test with None
        assert model._make_dataset(None) is None

        # Test with a numpy array
        data = np.array([1, 2, 3])
        result = model._make_dataset(data)
        assert isinstance(result, NDDataset)
        np.testing.assert_array_equal(result.data, data)

        # Test with an existing NDDataset
        dataset = NDDataset([4, 5, 6])
        result = model._make_dataset(dataset)
        assert isinstance(result, NDDataset)
        assert result is not dataset  # Should be a copy
        np.testing.assert_array_equal(result.data, dataset.data)

        # Test with a list of arrays
        data_list = [np.array([1, 2]), np.array([3, 4])]
        result = model._make_dataset(data_list)
        assert isinstance(result, list)
        assert all(isinstance(item, NDDataset) for item in result)

    def test_make_unsqueezed_dataset(self):
        """Test _make_unsqueezed_dataset method."""
        model = SimpleModel()

        # Test with 1D dataset
        dataset = NDDataset([1, 2, 3])
        result = model._make_unsqueezed_dataset(dataset)
        assert result.ndim == 2
        assert result.dims == ["y", "x"]
        np.testing.assert_array_equal(result.data, np.array([[1, 2, 3]]))

        # Test with masked 1D dataset
        dataset = NDDataset([1, 2, 3], mask=[False, True, False])
        result = model._make_unsqueezed_dataset(dataset)
        assert result.ndim == 2
        np.testing.assert_array_equal(result.mask, np.array([[False, True, False]]))

        # Test with 2D dataset (should remain unchanged)
        dataset = NDDataset([[1, 2], [3, 4]])
        result = model._make_unsqueezed_dataset(dataset)
        assert result.ndim == 2
        np.testing.assert_array_equal(result.data, dataset.data)

    def test_get_masked_rc(self):
        """Test _get_masked_rc method."""
        model = SimpleModel()
        model._X_shape = (3, 4)

        # Test with no mask
        mask = np.zeros((3, 4), dtype=bool)
        rows, cols = model._get_masked_rc(mask)
        np.testing.assert_array_equal(rows, np.zeros(3, dtype=bool))
        np.testing.assert_array_equal(cols, np.zeros(4, dtype=bool))

        # Test with masked rows
        mask = np.array(
            [
                [False, False, False, False],
                [True, True, True, True],
                [False, False, False, False],
            ]
        )
        rows, cols = model._get_masked_rc(mask)
        np.testing.assert_array_equal(rows, np.array([False, True, False]))
        np.testing.assert_array_equal(cols, np.zeros(4, dtype=bool))

        # Test with masked columns
        mask = np.array(
            [
                [False, True, False, True],
                [False, True, False, True],
                [False, True, False, True],
            ]
        )
        rows, cols = model._get_masked_rc(mask)
        np.testing.assert_array_equal(rows, np.zeros(3, dtype=bool))
        np.testing.assert_array_equal(cols, np.array([False, True, False, True]))

    def test_remove_masked_data(self):
        """Test _remove_masked_data method."""
        model = SimpleModel()

        # Test with no mask
        X = NDDataset([[1, 2], [3, 4]])
        result = model._remove_masked_data(X)
        assert result is X

        # Test with masked columns
        X = NDDataset(
            [[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [False, True, False]]
        )
        result = model._remove_masked_data(X)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result.data, np.array([[1, 3], [4, 6]]))
        assert not np.any(result._mask)  # Mask should be destroyed

        # Test with masked rows
        X = NDDataset(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            mask=[[False, False, False], [True, True, True], [False, False, False]],
        )
        result = model._remove_masked_data(X)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result.data, np.array([[1, 2, 3], [7, 8, 9]]))

    def test_restore_masked_data(self):
        """Test _restore_masked_data method."""
        model = SimpleModel()

        # Setup
        X_orig = NDDataset(
            [[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [False, True, False]]
        )
        model._X = X_orig  # This will trigger validation and preprocessing

        # Test restoring columns
        D = NDDataset([[1, 3], [4, 6]])
        result = model._restore_masked_data(D, axis=-1)
        assert result.shape == (2, 3)
        assert result.masked_data[0, 1] is MASKED
        assert result.masked_data[1, 1] is MASKED

        # Test restoring rows
        X_orig = NDDataset(
            [[1, 2], [3, 4], [5, 6]],
            mask=[[False, False], [True, True], [False, False]],
        )
        model._X = X_orig  # Fixed: was model.X (without underscore)
        D = NDDataset([[1, 2], [5, 6]])
        result = model._restore_masked_data(D, axis=-2)
        assert result.shape == (3, 2)
        assert result.masked_data[1, 0] is MASKED
        assert result.masked_data[1, 1] is MASKED

    def test_X_validation(self):
        """Test _X validation."""
        model = SimpleModel()

        # Test with a simple dataset
        X = NDDataset([[1, 2], [3, 4]])
        model._X = X
        assert model._X_shape == (2, 2)
        assert not np.any(model._X_mask)

        # Test with a masked dataset
        X = NDDataset([[1, 2], [3, 4]], mask=[[False, True], [False, False]])
        model._X = X
        assert model._X_shape == (2, 2)
        np.testing.assert_array_equal(model._X_mask, X.mask)

    def test_X_is_missing(self):
        """Test _X_is_missing property."""
        model = SimpleModel()

        # Initially X should be missing
        assert model._X_is_missing

        # After setting X, it should not be missing
        model._X = NDDataset([1, 2, 3])
        assert not model._X_is_missing

        # Setting X to None should make it missing again
        with pytest.raises(NotTransformedError):
            model._X = None

    def test_preprocess_as_X_changed(self):
        """Test _preprocess_as_X_changed method."""
        model = SimpleModel()
        X = NDDataset([[1, 2], [3, 4]])
        model._X = X

        # Check if _X_preprocessed was set correctly
        np.testing.assert_array_equal(model._X_preprocessed, X.data)

    def test_log_property(self):
        """Test log property."""
        model = SimpleModel()

        # The log property should return a string
        assert isinstance(model.log, str)
