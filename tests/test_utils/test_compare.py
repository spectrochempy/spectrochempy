# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np

from spectrochempy import NDDataset
from spectrochempy.utils.compare import dict_compare
from spectrochempy.utils.compare import difference


class TestDictCompare:
    def test_identical_dicts(self):
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"a": 1, "b": 2, "c": 3}

        # Test with check_equal_only=True
        assert dict_compare(d1, d2) is True

        # Test with check_equal_only=False
        added, removed, modified, same = dict_compare(d1, d2, check_equal_only=False)
        assert added == set()
        assert removed == set()
        assert modified == set()
        assert same == {"a", "b", "c"}

    def test_different_keys(self):
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"a": 1, "b": 2, "d": 4}

        # Test with check_equal_only=True
        assert dict_compare(d1, d2) is False

        # Test with check_equal_only=False
        added, removed, modified, same = dict_compare(d1, d2, check_equal_only=False)
        assert added == {"c"}
        assert removed == {"d"}
        assert modified == {"c", "d"}
        assert same == {"a", "b"}

    def test_modified_values(self):
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"a": 1, "b": 5, "c": 3}

        # Test with check_equal_only=True
        assert dict_compare(d1, d2) is False

        # Test with check_equal_only=False
        added, removed, modified, same = dict_compare(d1, d2, check_equal_only=False)
        assert added == set()
        assert removed == set()
        assert modified == {"b"}
        assert same == {"a", "c"}

    def test_sequence_values(self):
        d1 = {"a": [1, 2, 3], "b": 2}
        d2 = {"a": [1, 2, 3], "b": 2}
        d3 = {"a": [1, 2, 4], "b": 2}
        d4 = {"a": [1, 2], "b": 2}

        # Test with identical sequences
        assert dict_compare(d1, d2) is True

        # Test with different sequence values
        assert dict_compare(d1, d3) is False

        # Test with different sequence lengths
        assert dict_compare(d1, d4) is False

        # Test with array values
        d5 = {"a": np.array([1, 2, 3]), "b": 2}
        d6 = {"a": np.array([1, 2, 3]), "b": 2}
        d7 = {"a": np.array([1, 2, 4]), "b": 2}

        assert dict_compare(d5, d6) is True
        assert dict_compare(d5, d7) is False


class TestDifference:
    def test_difference_calculations(self):
        # Create sample NDDatasets
        x = NDDataset([1.0, 2.0, 3.0, 4.0])
        y = NDDataset([1.1, 1.9, 3.1, 3.9])

        max_abs_error, max_rel_error = difference(x, y)

        # Maximum absolute error should be 0.1
        assert np.isclose(max_abs_error, 0.1)

        # Maximum relative error (0.1/1.0 ≈ 0.1...)
        assert np.isclose(max_rel_error, 0.1 / 1.0)

    def test_with_zeros(self):
        # Test with zeros in y (should ignore these positions)
        x = NDDataset([1.0, 2.0, 3.0, 4.0])
        y = NDDataset([1.0, 0.0, 3.0, 4.0])

        max_abs_error, max_rel_error = difference(x, y)

        # Maximum absolute error should be 2.0 (at the zero position)
        assert np.isclose(max_abs_error, 2.0)

        # Maximum relative error should only consider non-zero positions
        # In non-zero positions, the maximum relative error is 0.0
        assert np.isclose(max_rel_error, 0.0)
