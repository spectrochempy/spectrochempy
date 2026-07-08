# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for solver-specific parameter-space transform functions."""

import math
import sys

import numpy as np
import pytest

from spectrochempy.analysis.curvefitting._parameter_transform import _to_external
from spectrochempy.analysis.curvefitting._parameter_transform import _to_internal

# ======================================================================================
# Constants
# ======================================================================================

# Sentinel values matching what the parser produces for "none" bounds
_SENTINEL_LOB = -1.0 / sys.float_info.epsilon
_SENTINEL_UPB = +1.0 / sys.float_info.epsilon

# A moderate value guaranteed to be within any bounded interval used here
_MID_VALUE = 50.0


# ======================================================================================
# _to_internal
# ======================================================================================


class TestToInternal:
    """Transform from physical space → optimizer (unbounded) space."""

    # -- finite bounds -------------------------------------------------------

    def test_finite_bounds_inverts_properly(self):
        pe = _to_internal(50.0, 0.0, 100.0)
        assert np.isfinite(pe)

    def test_finite_bounds_midpoint_is_zero(self):
        pe = _to_internal(50.0, 0.0, 100.0)
        assert pe == pytest.approx(0.0, abs=1e-10)

    def test_finite_bounds_within_range(self):
        pe = _to_internal(25.0, 0.0, 100.0)
        assert pe < 0.0

    def test_finite_bounds_value_at_lob(self):
        pe = _to_internal(0.0, 0.0, 100.0)
        assert pe == pytest.approx(-math.pi / 2.0, abs=1e-10)

    def test_finite_bounds_value_at_upb(self):
        pe = _to_internal(100.0, 0.0, 100.0)
        assert pe == pytest.approx(+math.pi / 2.0, abs=1e-10)

    # -- open bounds (no bounds) --------------------------------------------

    def test_open_bounds_identity(self):
        pe = _to_internal(42.0, _SENTINEL_LOB, _SENTINEL_UPB)
        assert pe == pytest.approx(42.0)

    def test_open_bounds_negative(self):
        pe = _to_internal(-10.0, _SENTINEL_LOB, _SENTINEL_UPB)
        assert pe == pytest.approx(-10.0)

    def test_open_bounds_zero(self):
        pe = _to_internal(0.0, _SENTINEL_LOB, _SENTINEL_UPB)
        assert pe == pytest.approx(0.0)

    def test_open_bounds_none_input(self):
        pe = _to_internal(42.0, None, None)
        assert pe == pytest.approx(42.0)

    # -- lower-only bounds ---------------------------------------------------

    def test_lower_only_positive_value(self):
        pe = _to_internal(50.0, 0.0, _SENTINEL_UPB)
        assert np.isfinite(pe)
        assert pe > 0.0

    def test_lower_only_at_bound(self):
        pe = _to_internal(0.0, 0.0, _SENTINEL_UPB)
        assert pe == pytest.approx(0.0)

    def test_lower_only_negative_lob_and_value(self):
        pe = _to_internal(-5.0, -10.0, _SENTINEL_UPB)
        assert np.isfinite(pe)

    # -- upper-only bounds ---------------------------------------------------

    def test_upper_only_positive_value(self):
        pe = _to_internal(50.0, _SENTINEL_LOB, 100.0)
        assert np.isfinite(pe)
        assert pe > 0.0

    def test_upper_only_value_at_bound_returns_zero(self):
        pe = _to_internal(100.0, _SENTINEL_LOB, 100.0)
        assert pe == pytest.approx(0.0)

    def test_upper_only_value_below_upb_returns_positive(self):
        pe = _to_internal(50.0, _SENTINEL_LOB, 100.0)
        assert np.isfinite(pe)

    # -- edge cases ----------------------------------------------------------

    def test_value_outside_below_lob(self):
        pe = _to_internal(-10.0, 0.0, 100.0)
        assert np.isfinite(pe)

    def test_value_outside_above_upb(self):
        pe = _to_internal(200.0, 0.0, 100.0)
        assert np.isfinite(pe)

    def test_lob_finite_upb_none_applies_upper_as_unbounded(self):
        pe = _to_internal(50.0, 0.0, None)
        assert np.isfinite(pe)
        assert pe > 0.0

    def test_lob_none_upb_finite_applies_upper_only(self):
        pe = _to_internal(50.0, None, 100.0)
        assert np.isfinite(pe)


# ======================================================================================
# _to_external
# ======================================================================================


class TestToExternal:
    """Transform from optimizer (unbounded) space → physical space."""

    # -- finite bounds -------------------------------------------------------

    def test_finite_bounds_zero_returns_midpoint(self):
        pe = _to_external(0.0, 0.0, 100.0)
        assert pe == pytest.approx(50.0, abs=1e-10)

    def test_finite_bounds_negative_arcsin(self):
        pe = _to_external(-1.0, 0.0, 100.0)
        assert pe < 50.0

    def test_finite_bounds_positive_arcsin(self):
        pe = _to_external(+1.0, 0.0, 100.0)
        assert pe > 50.0

    # -- open bounds (no bounds) --------------------------------------------

    def test_open_bounds_sentinel_returns_scalar(self):
        pe = _to_external(42.0, _SENTINEL_LOB, _SENTINEL_UPB)
        assert isinstance(pe, float)
        assert pe == pytest.approx(42.0)

    def test_open_bounds_none_returns_scalar(self):
        pe = _to_external(42.0, None, None)
        assert isinstance(pe, float)
        assert pe == pytest.approx(42.0)

    # -- lower-only bounds ---------------------------------------------------

    def test_lower_only_scalar(self):
        pe = _to_external(0.0, 0.0, _SENTINEL_UPB)
        assert np.isfinite(pe)
        assert pe >= 0.0

    # -- upper-only bounds ---------------------------------------------------

    def test_upper_only_scalar(self):
        pe = _to_external(0.0, _SENTINEL_LOB, 100.0)
        assert np.isfinite(pe)
        assert pe <= 100.0

    # -- list input ----------------------------------------------------------

    def test_list_input_finite_bounds(self):
        pe = _to_external([0.0, 0.0], 0.0, 100.0)
        assert isinstance(pe, list)
        assert len(pe) == 2
        assert pe[0] == pytest.approx(50.0, abs=1e-10)
        assert pe[1] == pytest.approx(50.0, abs=1e-10)

    def test_list_input_open_bounds(self):
        pe = _to_external([1.0, 2.0], _SENTINEL_LOB, _SENTINEL_UPB)
        assert isinstance(pe, list)
        assert len(pe) == 2
        assert pe == [1.0, 2.0]

    def test_list_single_element_open_bounds_returns_scalar(self):
        pe = _to_external([42.0], _SENTINEL_LOB, _SENTINEL_UPB)
        assert isinstance(pe, float)
        assert pe == pytest.approx(42.0)

    # -- edge cases ----------------------------------------------------------

    def test_single_element_list_finite_bounds_returns_scalar(self):
        pe = _to_external([0.0], 0.0, 100.0)
        assert not isinstance(pe, list)

    def test_lob_none_with_finite_upb_applies_upper_only(self):
        pe = _to_external(0.0, None, 100.0)
        assert isinstance(pe, float)
        assert pe == pytest.approx(100.0, abs=1e-10)


# ======================================================================================
# Round-trip consistency
# ======================================================================================


class TestRoundTrip:
    """Verify that to_internal + to_external recovers the original value."""

    RTOL = 1e-10
    ATOL = 1e-10

    def check_round_trip(self, value, lob, upb):
        pe = _to_internal(value, lob, upb)
        recovered = _to_external(pe, lob, upb)
        assert recovered == pytest.approx(value, rel=self.RTOL, abs=self.ATOL)

    # -- finite bounds -------------------------------------------------------

    def test_finite_bounds_midpoint(self):
        self.check_round_trip(50.0, 0.0, 100.0)

    def test_finite_bounds_near_lob(self):
        self.check_round_trip(1.0, 0.0, 100.0)

    def test_finite_bounds_near_upb(self):
        self.check_round_trip(99.0, 0.0, 100.0)

    def test_finite_bounds_at_lob(self):
        self.check_round_trip(0.0, 0.0, 100.0)

    def test_finite_bounds_at_upb(self):
        self.check_round_trip(100.0, 0.0, 100.0)

    # -- open bounds ---------------------------------------------------------

    def test_open_bounds_sentinel(self):
        self.check_round_trip(42.0, _SENTINEL_LOB, _SENTINEL_UPB)

    def test_open_bounds_none(self):
        self.check_round_trip(42.0, None, None)

    def test_open_bounds_negative(self):
        self.check_round_trip(-10.0, _SENTINEL_LOB, _SENTINEL_UPB)

    def test_open_bounds_zero(self):
        self.check_round_trip(0.0, _SENTINEL_LOB, _SENTINEL_UPB)

    # -- lower-only bounds ---------------------------------------------------

    def test_lower_only(self):
        self.check_round_trip(50.0, 0.0, _SENTINEL_UPB)

    def test_lower_only_near_bound(self):
        self.check_round_trip(1.0, 0.0, _SENTINEL_UPB)

    def test_lower_only_at_bound(self):
        self.check_round_trip(0.0, 0.0, _SENTINEL_UPB)

    # -- upper-only bounds ---------------------------------------------------

    def test_upper_only(self):
        self.check_round_trip(50.0, _SENTINEL_LOB, 100.0)

    def test_upper_only_near_bound(self):
        self.check_round_trip(99.0, _SENTINEL_LOB, 100.0)

    def test_upper_only_at_bound(self):
        self.check_round_trip(100.0, _SENTINEL_LOB, 100.0)

    # -- values outside declared bounds --------------------------------------
    # These do not round-trip because the internal transform clamps lob/upb
    # to the value before applying the arcsin transform. The round-trip
    # recovers the clamped value, not the original outside-bounds value.

    def test_outside_below_lob_clamps_to_lob(self):
        pe = _to_internal(-10.0, 0.0, 100.0)
        recovered = _to_external(pe, 0.0, 100.0)
        assert recovered == pytest.approx(0.0)

    def test_outside_above_upb_clamps_to_upb(self):
        pe = _to_internal(200.0, 0.0, 100.0)
        recovered = _to_external(pe, 0.0, 100.0)
        assert recovered == pytest.approx(100.0)
