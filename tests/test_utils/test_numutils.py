# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

from spectrochempy.utils.numutils import get_n_decimals
from spectrochempy.utils.numutils import gt_eps
from spectrochempy.utils.numutils import largest_power_of_2
from spectrochempy.utils.numutils import spacings


class TestGetNDecimals:
    def test_positive_number(self):
        assert get_n_decimals(123.456, sigdigits=3) == 0
        assert get_n_decimals(12.345, sigdigits=3) == 1
        assert get_n_decimals(1.2345, sigdigits=3) == 2
        assert get_n_decimals(0.12345, sigdigits=3) == 3
        assert get_n_decimals(0.012345, sigdigits=3) == 4

    def test_negative_number(self):
        assert get_n_decimals(-123.456, sigdigits=3) == 0
        assert get_n_decimals(-1.2345, sigdigits=3) == 2

    def test_different_sigdigits(self):
        assert get_n_decimals(123.456, sigdigits=4) == 1
        assert get_n_decimals(123.456, sigdigits=5) == 2
        assert get_n_decimals(123.456, sigdigits=6) == 3

    def test_overflow(self):
        # Testing very small numbers that might cause overflow
        assert get_n_decimals(1e-500) == 2  # OverflowError case


class TestSpacings:
    def test_uniform_spacing(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert spacings(arr) == 1.0

        arr = np.array([0, 0.1, 0.2, 0.3])
        assert spacings(arr) == 0.1

    def test_non_uniform_spacing(self):
        arr = np.array([1, 2, 4, 7])
        result = spacings(arr)
        assert isinstance(result, list)
        assert 1.0 in result and 2.0 in result and 3.0 in result

    def test_edge_cases(self):
        with pytest.raises(ValueError):
            spacings(np.array([]))

        assert spacings(np.array([5])) == 0

    def test_significant_digits(self):
        arr = np.array([0, 0.1001, 0.2002, 0.3003])
        assert spacings(arr, sd=3) == 0.1

        # Create an array with actually different spacings at higher precision
        arr_diff = np.array([0, 0.1001, 0.2005, 0.3010])
        assert isinstance(
            spacings(arr_diff, sd=5), list
        )  # With higher precision, spacing is not uniform


class TestGtEps:
    def test_greater_than_epsilon(self):
        from spectrochempy.utils.constants import EPSILON

        arr = np.array([EPSILON * 2, EPSILON * 3])
        assert gt_eps(arr)

    def test_not_greater_than_epsilon(self):
        from spectrochempy.utils.constants import EPSILON

        arr = np.array([EPSILON * 0.5, EPSILON * 0.9])
        assert not gt_eps(arr)

    def test_mixed_values(self):
        from spectrochempy.utils.constants import EPSILON

        arr = np.array([EPSILON * 0.5, EPSILON * 2])
        assert gt_eps(arr)


class TestLargestPowerOf2:
    def test_exact_powers(self):
        assert largest_power_of_2(1) == 1
        assert largest_power_of_2(2) == 2
        assert largest_power_of_2(4) == 4
        assert largest_power_of_2(8) == 8
        assert largest_power_of_2(16) == 16

    def test_non_exact_powers(self):
        assert largest_power_of_2(3) == 4
        assert largest_power_of_2(5) == 8
        assert largest_power_of_2(10) == 16
        assert largest_power_of_2(33) == 64

    def test_large_values(self):
        assert largest_power_of_2(1023) == 1024
        assert largest_power_of_2(1025) == 2048
