"""
Tests for coordinate-aware Savitzky-Golay derivatives (PR 2, issue #1091).

Covers:
- analytic accuracy on y = 3x² (deriv=1 → 6x, deriv=2 → 6)
- ascending / descending uniform coordinates
- unit-independent sign (cm⁻¹, ppm, unitless)
- delta=None auto-detect vs explicit delta
- irregular coordinate warning
- missing / degenerate / non-finite / non-numeric coordinate fallback
- deriv=0 invariance
- invariants (non-mutation, shape, coords, title)
- API equivalence after PR 1 dim fix
- Filter object reuse (no delta leak)
- explicit delta + _reversed interaction (cm⁻¹/ppm)
"""

import warnings

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.processing.filter.filter import Filter
from spectrochempy.processing.filter.filter import _detect_uniform_spacing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H = 0.18367346938775511  # (10-1)/49  — spacing of linspace(1, 10, 50)


def _make_2d(data_1d, coord_values, unit=None, title="wavenumber"):
    ds = NDDataset(data_1d.reshape(1, -1))
    c = Coord(coord_values, title=title)
    if unit:
        c.units = unit
    ds.set_coordset(x=c)
    return ds


@pytest.fixture
def ds_asc():
    """Ascending uniform coordinate, y = 3x², no unit."""
    x = np.linspace(1, 10, 50)
    return _make_2d(3.0 * x**2, x, unit=None, title="x"), x


@pytest.fixture
def ds_desc():
    """Descending uniform coordinate, y = 3x², no unit."""
    x = np.linspace(10, 1, 50)
    return _make_2d(3.0 * x**2, x, unit=None, title="x"), x


@pytest.fixture
def ds_asc_cm1():
    """Ascending uniform coordinate with wavenumber unit."""
    x = np.linspace(1, 10, 50)
    return _make_2d(3.0 * x**2, x, unit="1/centimeter"), x


@pytest.fixture
def ds_desc_cm1():
    """Descending uniform coordinate with wavenumber unit."""
    x = np.linspace(10, 1, 50)
    return _make_2d(3.0 * x**2, x, unit="1/centimeter"), x


@pytest.fixture
def ds_asc_ppm():
    """Ascending uniform coordinate with ppm unit."""
    x = np.linspace(1, 10, 50)
    return _make_2d(3.0 * x**2, x, unit="ppm"), x


@pytest.fixture
def ds_irreg():
    """Irregular coordinate."""
    x = np.sort(np.random.default_rng(42).uniform(1, 10, 50))
    return _make_2d(3.0 * x**2, x, title="x"), x


MID = 25  # midpoint index for 50-point arrays (away from edges)


# ---------------------------------------------------------------------------
# Analytic accuracy — deriv=1
# ---------------------------------------------------------------------------


class TestDerivative1Analytic:
    """y = 3x² → y' = 6x."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["ds_asc", "ds_desc", "ds_asc_cm1", "ds_desc_cm1", "ds_asc_ppm"],
    )
    def test_deriv1_auto_delta(self, fixture_name, request):
        ds, x = request.getfixturevalue(fixture_name)
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        expected = 6.0 * x[MID]
        actual = float(r.data[0, MID])
        # With _data precision the auto-detected delta is exact to ~1e-14.
        assert (
            abs(actual - expected) < 1e-12
        ), f"{fixture_name}: deriv=1 auto: actual={actual:.10f}, expected={expected:.10f}"

    def test_deriv1_sign_ascending(self, ds_asc):
        ds, x = ds_asc
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert r.data[0, MID] > 0, "ascending deriv=1 should be positive"

    def test_deriv1_sign_descending(self, ds_desc):
        ds, x = ds_desc
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert r.data[0, MID] > 0, "descending deriv=1 should be positive (6x > 0)"

    def test_deriv1_sign_cm1_ascending(self, ds_asc_cm1):
        ds, x = ds_asc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert r.data[0, MID] > 0, "ascending cm-1 deriv=1 should be positive"

    def test_deriv1_sign_cm1_descending(self, ds_desc_cm1):
        ds, x = ds_desc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert r.data[0, MID] > 0, "descending cm-1 deriv=1 should be positive"

    def test_deriv1_explicit_delta_matches_scipy(self, ds_asc):
        ds, x = ds_asc
        r = scp.savgol(ds, size=7, order=3, deriv=1, delta=H)
        sp = Filter(method="savgol", size=7, order=3, deriv=1, delta=H).transform(ds)
        np.testing.assert_allclose(r.data, sp.data, atol=1e-14)


# ---------------------------------------------------------------------------
# Analytic accuracy — deriv=2
# ---------------------------------------------------------------------------


class TestDerivative2Analytic:
    """y = 3x² → y'' = 6."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["ds_asc", "ds_desc", "ds_asc_cm1", "ds_desc_cm1", "ds_asc_ppm"],
    )
    def test_deriv2_auto_delta(self, fixture_name, request):
        ds, x = request.getfixturevalue(fixture_name)
        r = scp.savgol(ds, size=7, order=3, deriv=2)
        actual = float(r.data[0, MID])
        assert (
            abs(actual - 6.0) < 1e-10
        ), f"{fixture_name}: deriv=2 auto: actual={actual:.10f}, expected=6.0000000000"


# ---------------------------------------------------------------------------
# deriv=0 invariance
# ---------------------------------------------------------------------------


class TestDeriv0Invariance:
    """deriv=0 must be bit-identical to the input (no delta detection triggered)."""

    def test_deriv0_unchanged(self, ds_asc):
        ds, x = ds_asc
        r = scp.savgol(ds, size=7, order=3, deriv=0)
        np.testing.assert_allclose(r.data, ds.data, atol=1e-14)

    def test_deriv0_no_warning(self, ds_asc):
        ds, x = ds_asc
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=0)
            coord_warnings = [
                x for x in w if "uniformly spaced" in str(x.message).lower()
            ]
            assert len(coord_warnings) == 0


# ---------------------------------------------------------------------------
# Delta priority
# ---------------------------------------------------------------------------


class TestDeltaPriority:
    """Explicit delta must take precedence over auto-detection."""

    def test_explicit_delta_overrides_auto(self, ds_asc):
        ds, x = ds_asc
        r_auto = scp.savgol(ds, size=7, order=3, deriv=1)
        r_explicit = scp.savgol(ds, size=7, order=3, deriv=1, delta=1.0)
        # auto uses h, explicit uses 1.0 → different numerical results
        assert not np.allclose(r_auto.data, r_explicit.data, atol=1e-6)

    def test_explicit_delta_h_matches_scipy(self, ds_asc):
        ds, x = ds_asc
        r = scp.savgol(ds, size=7, order=3, deriv=1, delta=H)
        import scipy.signal

        expected = scipy.signal.savgol_filter(ds.data, 7, 3, axis=-1, deriv=1, delta=H)
        np.testing.assert_allclose(r.data, expected, atol=1e-14)

    def test_filter_reuse_no_delta_leak(self):
        """Single Filter object reused on two datasets with different coords."""
        x1 = np.linspace(1, 10, 50)
        x2 = np.linspace(1, 20, 50)
        ds1 = _make_2d(3.0 * x1**2, x1)
        ds2 = _make_2d(3.0 * x2**2, x2)

        f = Filter(method="savgol", size=7, order=3, deriv=1)
        r1 = f.transform(ds1)
        r2 = f.transform(ds2)

        expected1 = 6.0 * x1[MID]
        expected2 = 6.0 * x2[MID]
        assert (
            abs(float(r1.data[0, MID]) - expected1) < 1e-12
        ), f"ds1: actual={float(r1.data[0,MID]):.6f}, expected={expected1:.6f}"
        assert (
            abs(float(r2.data[0, MID]) - expected2) < 1e-12
        ), f"ds2: actual={float(r2.data[0,MID]):.6f}, expected={expected2:.6f}"


# ---------------------------------------------------------------------------
# Irregular coordinate
# ---------------------------------------------------------------------------


class TestIrregularCoordinate:
    """Non-uniform coordinate must trigger a warning and fall back to delta=1.0."""

    def test_warning_emitted(self, ds_irreg):
        ds, x = ds_irreg
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=1)
            coord_warnings = [
                x for x in w if "uniformly spaced" in str(x.message).lower()
            ]
            assert len(coord_warnings) >= 1

    def test_fallback_to_index_based(self, ds_irreg):
        ds, x = ds_irreg
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            r_auto = scp.savgol(ds, size=7, order=3, deriv=1)
        r_explicit = scp.savgol(ds, size=7, order=3, deriv=1, delta=1.0)
        np.testing.assert_allclose(r_auto.data, r_explicit.data, atol=1e-14)

    def test_explicit_delta_suppresses_warning(self, ds_irreg):
        ds, x = ds_irreg
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=1, delta=1.0)
            coord_warnings = [
                x for x in w if "uniformly spaced" in str(x.message).lower()
            ]
            assert len(coord_warnings) == 0


# ---------------------------------------------------------------------------
# Missing / degenerate / non-finite / non-numeric coordinate
# ---------------------------------------------------------------------------


class TestEdgeCoordinateCases:
    """Missing, degenerate, non-finite, or non-numeric coordinates trigger fallback."""

    def test_no_coord(self):
        ds = NDDataset(np.random.rand(1, 50))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=1)
            coord_warnings = [x for x in w if "falling back" in str(x.message).lower()]
            assert len(coord_warnings) >= 1

    def test_degenerate_coord(self):
        ds = NDDataset(np.random.rand(1, 50))
        ds.set_coordset(x=Coord(np.ones(50), title="x"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=1)
            coord_warnings = [x for x in w if "degenerate" in str(x.message).lower()]
            assert len(coord_warnings) >= 1

    def test_nan_in_coord(self):
        x = np.linspace(1, 10, 50)
        x[25] = np.nan
        ds = NDDataset(np.random.rand(1, 50))
        ds.set_coordset(x=Coord(x, title="x"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=1)
            coord_warnings = [x for x in w if "non-finite" in str(x.message).lower()]
            assert len(coord_warnings) >= 1

    def test_repeated_values_giving_zero_diff(self):
        x = np.ones(50)
        x[0] = 1.0
        x[-1] = 2.0
        ds = NDDataset(np.random.rand(1, 50))
        ds.set_coordset(x=Coord(x, title="x"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scp.savgol(ds, size=7, order=3, deriv=1)
            # non-uniform → must warn about falling back
            coord_warnings = [x for x in w if "falling back" in str(x.message).lower()]
            assert len(coord_warnings) >= 1

    def test_non_numeric_coord(self):
        """
        A non-numeric coord triggers a fallback warning.

        String coords cannot be created through the normal Coord API, so
        this tests the helper directly with a mock whose _data is string.
        """
        from unittest.mock import MagicMock

        mock_ds = MagicMock()
        mock_coord = MagicMock()
        mock_coord.is_masked = False
        mock_coord._mask = False
        mock_coord._data = np.array(["a", "b", "c", "d", "e"])
        mock_ds.coord.return_value = mock_coord
        delta, msg = _detect_uniform_spacing(mock_ds, -1)
        assert delta is None
        assert "not numeric" in msg.lower()


# ---------------------------------------------------------------------------
# Explicit delta + _reversed interaction (cm⁻¹/ppm)
# ---------------------------------------------------------------------------


class TestExplicitDeltaReversed:
    """
    Explicit delta + _reversed interaction documents a known bug.

    For cm⁻¹/ppm coords with explicit positive delta, _reversed flips
    the sign of odd-order derivatives.  xfail(strict=True) ensures the
    test fails if the bug is ever fixed, prompting an update.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="Explicit delta with cm-1 ascending: _reversed incorrectly "
        "flips sign. Expected positive. See #1091 follow-up.",
    )
    def test_explicit_delta_cm1_ascending_positive(self, ds_asc_cm1):
        ds, x = ds_asc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1, delta=H)
        # Currently returns negative due to _reversed; correct is positive
        assert float(r.data[0, MID]) > 0

    def test_explicit_delta_cm1_descending_correct(self, ds_desc_cm1):
        """Explicit delta=H with descending cm-1 → _reversed cancels correctly."""
        ds, x = ds_desc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1, delta=H)
        expected = 6.0 * x[MID]
        # For descending coords, _reversed happens to correct the sign
        # and the magnitude matches.
        assert float(r.data[0, MID]) > 0
        assert abs(float(r.data[0, MID]) - expected) < 1e-12

    @pytest.mark.xfail(
        strict=True,
        reason="Explicit negative delta with cm-1 ascending: _reversed "
        "flips sign. Expected negative. See #1091 follow-up.",
    )
    def test_explicit_delta_cm1_negative(self, ds_asc_cm1):
        ds, x = ds_asc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1, delta=-H)
        # Correct result: negative (descending direction)
        assert float(r.data[0, MID]) < 0

    def test_auto_delta_cm1_ascending_positive(self, ds_asc_cm1):
        """Auto-detected delta with ascending cm-1 → correct positive sign."""
        ds, x = ds_asc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert float(r.data[0, MID]) > 0

    def test_auto_delta_cm1_descending_positive(self, ds_desc_cm1):
        """Auto-detected delta with descending cm-1 → correct positive sign."""
        ds, x = ds_desc_cm1
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        # descending coord → negative auto-delta → y=3x², 6x > 0
        assert float(r.data[0, MID]) > 0


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Non-mutation, shape, coords, title preservation."""

    def test_source_not_mutated(self, ds_asc):
        ds, x = ds_asc
        original_data = ds.data.copy()
        scp.savgol(ds, size=7, order=3, deriv=1)
        np.testing.assert_array_equal(ds.data, original_data)

    def test_coord_not_mutated(self, ds_asc):
        ds, x = ds_asc
        original_coord = ds.coord(-1).data.copy()
        scp.savgol(ds, size=7, order=3, deriv=1)
        np.testing.assert_array_equal(ds.coord(-1).data, original_coord)

    def test_shape_preserved(self, ds_asc):
        ds, x = ds_asc
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert r.shape == ds.shape

    def test_dims_preserved(self, ds_asc):
        ds, x = ds_asc
        r = scp.savgol(ds, size=7, order=3, deriv=1)
        assert r.dims == ds.dims


# ---------------------------------------------------------------------------
# API equivalence
# ---------------------------------------------------------------------------


class TestAPIEquivalence:
    """All entry points produce identical results for coordinate-aware path."""

    def test_function_method_alias(self, ds_asc):
        ds, x = ds_asc
        r_func = scp.savgol(ds, size=7, order=3, deriv=1)
        r_method = ds.savgol(size=7, order=3, deriv=1)
        r_alias = scp.savgol_filter(ds, size=7, order=3, deriv=1)
        np.testing.assert_allclose(r_func.data, r_method.data, atol=1e-14)
        np.testing.assert_allclose(r_func.data, r_alias.data, atol=1e-14)

    def test_transformer_default_delta(self, ds_asc):
        """Filter(method='savgol') with default delta=None must match savgol()."""
        ds, x = ds_asc
        r_func = scp.savgol(ds, size=7, order=3, deriv=1)
        r_trans = Filter(method="savgol", size=7, order=3, deriv=1).transform(ds)
        np.testing.assert_allclose(r_func.data, r_trans.data, atol=1e-14)

    def test_dim_string(self, ds_asc):
        ds, x = ds_asc
        r_int = scp.savgol(ds, size=7, order=3, deriv=1, dim=-1)
        r_str = scp.savgol(ds, size=7, order=3, deriv=1, dim="x")
        np.testing.assert_allclose(r_int.data, r_str.data, atol=1e-14)


# ---------------------------------------------------------------------------
# _detect_uniform_spacing helper
# ---------------------------------------------------------------------------


class TestDetectUniformSpacing:
    """Direct tests on the module-level helper function."""

    def test_uniform_ascending(self):
        x = np.linspace(1, 10, 50)
        ds = _make_2d(np.ones(50), x)
        delta, msg = _detect_uniform_spacing(ds, -1)
        assert delta is not None
        assert delta > 0
        assert abs(delta - H) < 1e-14
        assert msg is None

    def test_uniform_descending(self):
        x = np.linspace(10, 1, 50)
        ds = _make_2d(np.ones(50), x)
        delta, msg = _detect_uniform_spacing(ds, -1)
        assert delta is not None
        assert delta < 0
        assert abs(delta - (-H)) < 1e-14
        assert msg is None

    def test_returns_mean_not_first(self):
        """Return value must be the mean signed delta, not diffs[0]."""
        x = np.linspace(1, 10, 50)
        ds = _make_2d(np.ones(50), x)
        delta, msg = _detect_uniform_spacing(ds, -1)
        # mean of raw diffs = H exactly
        assert delta == pytest.approx(H, abs=1e-14)

    def test_irregular(self):
        x = np.sort(np.random.default_rng(42).uniform(1, 10, 50))
        ds = _make_2d(np.ones(50), x)
        delta, msg = _detect_uniform_spacing(ds, -1)
        assert delta is None
        assert "not uniformly spaced" in msg

    def test_no_coord(self):
        ds = NDDataset(np.ones((1, 50)))
        delta, msg = _detect_uniform_spacing(ds, -1)
        assert delta is None
        assert "no coordinate" in msg.lower() or "none" in msg.lower()

    def test_nan_coord(self):
        x = np.linspace(1, 10, 50)
        x[10] = np.nan
        ds = _make_2d(np.ones(50), x)
        delta, msg = _detect_uniform_spacing(ds, -1)
        assert delta is None
        assert "non-finite" in msg.lower()

    def test_single_point_coord(self):
        ds = NDDataset(np.ones((1, 1)))
        ds.set_coordset(x=Coord(np.array([5.0]), title="x"))
        delta, msg = _detect_uniform_spacing(ds, -1)
        assert delta is None
        assert "fewer than 2" in msg.lower()

    def test_non_numeric_coord(self):
        from unittest.mock import MagicMock

        mock_ds = MagicMock()
        mock_coord = MagicMock()
        mock_coord.is_masked = False
        mock_coord._mask = False
        mock_coord._data = np.array(["a", "b", "c", "d", "e"])
        mock_ds.coord.return_value = mock_coord
        delta, msg = _detect_uniform_spacing(mock_ds, -1)
        assert delta is None
        assert "not numeric" in msg.lower()

    def test_masked_coord(self):
        """A coordinate with masked values triggers fallback."""
        from unittest.mock import MagicMock

        mock_ds = MagicMock()
        mock_coord = MagicMock()
        mock_coord.is_masked = True
        mock_coord._mask = np.array([False, False, True, False, False])
        mock_ds.coord.return_value = mock_coord
        delta, msg = _detect_uniform_spacing(mock_ds, -1)
        assert delta is None
        assert "masked" in msg.lower()

    def test_data_precision_vs_raw(self):
        """
        coord.data truncates float64; coord._data preserves it.

        This test justifies using the private _data attribute:
        the public .data property rounds values to ~4 significant digits,
        which destroys the uniform-spacing signal for linspace coords.
        """
        x = np.linspace(1, 10, 50)
        c = Coord(x, title="x")
        data_rounded = np.asarray(c.data, dtype=float)
        data_raw = np.asarray(c._data, dtype=float)
        max_err_rounded = np.max(np.abs(data_rounded - x))
        max_err_raw = np.max(np.abs(data_raw - x))
        # rounded has ~0.5% error, raw has machine-precision error
        assert max_err_rounded > 1e-4, "expected coord.data to truncate"
        assert max_err_raw < 1e-14, "expected coord._data to preserve precision"
