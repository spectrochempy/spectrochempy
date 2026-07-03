# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for preprocessing operations."""

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy import MASKED
from spectrochempy.processing.transformation.preprocessing import (
    autoscale,
    center,
    log_transform,
    msc,
    normalize,
    pareto_scale,
    range_scale,
    robust_scale,
    snv,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    AutoscaleTransformer,
    CenterTransformer,
    SNVTransformer,
)
from spectrochempy.utils.exceptions import SpectroChemPyError


@pytest.fixture
def simple_2d():
    """Return a small synthetic 2-D dataset for unit tests."""
    x = Coord(np.linspace(0, 5, 6), title="wavenumber")
    y = Coord(np.linspace(0, 3, 4), title="time")
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    return NDDataset(data, coordset=[y, x])


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_max(self, simple_2d):
        nd = normalize(simple_2d, method="max", dim="x")
        expected = simple_2d.data[0] / 6.0
        assert np.allclose(nd.data[0], expected)
        assert "normalize (max) applied on dimension x" in nd.history[-1].lower()

    def test_sum(self, simple_2d):
        nd = normalize(simple_2d, method="sum", dim="x")
        expected = simple_2d.data[0] / 21.0
        assert np.allclose(nd.data[0], expected)

    def test_vector(self, simple_2d):
        nd = normalize(simple_2d, method="vector", dim="x")
        norm = np.sqrt(np.sum(simple_2d.data[0] ** 2))
        assert np.allclose(nd.data[0], simple_2d.data[0] / norm)

    def test_minmax(self, simple_2d):
        nd = normalize(simple_2d, method="minmax", dim="x")
        expected = (simple_2d.data[0] - 1.0) / 5.0
        assert np.allclose(nd.data[0], expected)

    def test_inplace(self, simple_2d):
        nd = simple_2d.copy()
        result = normalize(nd, method="max", dim="x", inplace=True)
        assert result is nd
        assert np.allclose(nd.data[0], simple_2d.data[0] / 6.0)

    def test_unknown_method(self, simple_2d):
        with pytest.raises(SpectroChemPyError, match="Unknown normalization method"):
            normalize(simple_2d, method="unknown")

    def test_zero_norm(self, simple_2d):
        """Constant spectrum: max=0 must not raise."""
        nd = normalize(simple_2d, method="max", dim="x")
        # last spectrum is constant 1.0, so max is 1.0, not zero.
        # Create a true zero spectrum
        zero_data = np.zeros_like(simple_2d.data)
        zero_ds = NDDataset(zero_data, coordset=simple_2d.coordset)
        nd = normalize(zero_ds, method="max", dim="x")
        assert np.allclose(nd.data, 0.0)

    def test_units_preserved(self, simple_2d):
        simple_2d.units = "absorbance"
        nd = normalize(simple_2d, method="max", dim="x")
        assert nd.units == simple_2d.units


# ---------------------------------------------------------------------------
# center
# ---------------------------------------------------------------------------


class TestCenter:
    def test_default_dim_y(self, simple_2d):
        nd = center(simple_2d)
        expected = simple_2d.data[:, 0] - np.mean(simple_2d.data[:, 0])
        assert np.allclose(nd.data[:, 0], expected)
        assert "center applied on dimension y" in nd.history[-1].lower()

    def test_dim_x(self, simple_2d):
        nd = center(simple_2d, dim="x")
        expected = simple_2d.data[0] - np.mean(simple_2d.data[0])
        assert np.allclose(nd.data[0], expected)

    def test_inplace(self, simple_2d):
        nd = simple_2d.copy()
        result = center(nd, dim="x", inplace=True)
        assert result is nd


# ---------------------------------------------------------------------------
# autoscale
# ---------------------------------------------------------------------------


class TestAutoscale:
    def test_default_dim_y(self, simple_2d):
        nd = autoscale(simple_2d)
        col = simple_2d.data[:, 0]
        expected = (col - np.mean(col)) / np.std(col)
        assert np.allclose(nd.data[:, 0], expected)
        assert "autoscale applied on dimension y" in nd.history[-1].lower()

    def test_dim_x(self, simple_2d):
        nd = autoscale(simple_2d, dim="x")
        row = simple_2d.data[0]
        expected = (row - np.mean(row)) / np.std(row)
        assert np.allclose(nd.data[0], expected)

    def test_zero_std(self, simple_2d):
        """Constant column should yield zeros, not NaN."""
        const_data = np.ones((4, 6))
        const_ds = NDDataset(const_data, coordset=simple_2d.coordset)
        nd = autoscale(const_ds, dim="y")
        assert np.allclose(nd.data, 0.0)

    def test_inplace(self, simple_2d):
        nd = simple_2d.copy()
        result = autoscale(nd, dim="x", inplace=True)
        assert result is nd


# ---------------------------------------------------------------------------
# snv
# ---------------------------------------------------------------------------


class TestSNV:
    def test_equivalent_to_autoscale_dim_x(self, simple_2d):
        nd_snv = snv(simple_2d)
        nd_as = autoscale(simple_2d, dim="x")
        assert np.allclose(nd_snv.data, nd_as.data)
        assert "snv applied" in nd_snv.history[-1].lower()

    def test_inplace(self, simple_2d):
        nd = simple_2d.copy()
        result = snv(nd, inplace=True)
        assert result is nd


# ---------------------------------------------------------------------------
# msc
# ---------------------------------------------------------------------------


class TestMSC:
    def test_default_reference(self, simple_2d):
        nd = msc(simple_2d)
        ref = np.mean(simple_2d.data, axis=0)
        # Non-constant spectra should match the reference after correction
        for i in range(simple_2d.shape[0] - 1):
            assert np.allclose(nd.data[i], ref)
        # Constant spectrum (last row) has b=0, so it becomes zero
        assert np.allclose(nd.data[-1], 0.0)

    def test_explicit_reference(self, simple_2d):
        ref = simple_2d.data[0]
        nd = msc(simple_2d, reference=ref)
        # Spectrum 0 corrected against itself should equal itself
        assert np.allclose(nd.data[0], ref)

    def test_regression_against_ref(self, simple_2d):
        nd = msc(simple_2d)
        ref = np.mean(simple_2d.data, axis=0)
        # Non-constant spectra should regress to a≈0, b≈1
        for i in range(simple_2d.shape[0] - 1):
            # Fit corrected[i] = a + b * ref; should yield a≈0, b≈1
            n = ref.size
            sum_ref = np.sum(ref)
            sum_ref2 = np.sum(ref**2)
            den = n * sum_ref2 - sum_ref**2
            sum_corr = np.sum(nd.data[i])
            sum_corr_ref = np.sum(nd.data[i] * ref)
            b = (n * sum_corr_ref - sum_ref * sum_corr) / den
            a = (sum_corr - b * sum_ref) / n
            assert np.isclose(a, 0.0, atol=1e-10)
            assert np.isclose(b, 1.0, atol=1e-10)
        # Constant spectrum has b=0, so it becomes zero and does not regress
        assert np.allclose(nd.data[-1], 0.0)

    def test_constant_reference_error(self, simple_2d):
        const_ref = np.ones(simple_2d.shape[1])
        with pytest.raises(SpectroChemPyError, match="denominator is zero"):
            msc(simple_2d, reference=const_ref)

    def test_wrong_reference_size(self, simple_2d):
        with pytest.raises(SpectroChemPyError, match="reference size"):
            msc(simple_2d, reference=np.ones(3))

    def test_nd_error(self, simple_2d):
        """MSC currently supports only 2-D datasets."""
        data_3d = np.ones((2, 4, 6))
        ds_3d = NDDataset(data_3d)
        with pytest.raises(SpectroChemPyError, match="2-D datasets"):
            msc(ds_3d)

    def test_inplace(self, simple_2d):
        nd = simple_2d.copy()
        result = msc(nd, inplace=True)
        assert result is nd


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


class TestMasks:
    def test_mask_preserved_normalize(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = normalize(ds, method="max", dim="x")
        assert nd.mask[:, 2].all()

    def test_mask_preserved_center(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = center(ds, dim="y")
        assert nd.mask[:, 2].all()

    def test_mask_preserved_autoscale(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = autoscale(ds, dim="y")
        assert nd.mask[:, 2].all()

    def test_mask_preserved_snv(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = snv(ds)
        assert nd.mask[:, 2].all()

    def test_mask_preserved_msc(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = msc(ds)
        assert nd.mask[:, 2].all()

    def test_mask_preserved_pareto(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = pareto_scale(ds, dim="y")
        assert nd.mask[:, 2].all()

    def test_mask_preserved_range(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = range_scale(ds, dim="y")
        assert nd.mask[:, 2].all()

    def test_mask_preserved_robust(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = robust_scale(ds, dim="y")
        assert nd.mask[:, 2].all()

    def test_mask_preserved_log(self, simple_2d):
        ds = simple_2d.copy()
        ds[:, 2] = MASKED
        nd = log_transform(ds, method="log1p")
        assert nd.mask[:, 2].all()


# ---------------------------------------------------------------------------
# pareto_scale
# ---------------------------------------------------------------------------


class TestParetoScale:
    def test_basic(self, simple_2d):
        nd = pareto_scale(simple_2d, dim="y")
        col = simple_2d.data[:, 0]
        expected = (col - np.mean(col)) / np.sqrt(np.std(col))
        assert np.allclose(nd.data[:, 0], expected)

    def test_dim_x(self, simple_2d):
        nd = pareto_scale(simple_2d, dim="x")
        row = simple_2d.data[0]
        expected = (row - np.mean(row)) / np.sqrt(np.std(row))
        assert np.allclose(nd.data[0], expected)

    def test_zero_std(self, simple_2d):
        const_data = np.ones((4, 6))
        const_ds = NDDataset(const_data, coordset=simple_2d.coordset)
        nd = pareto_scale(const_ds, dim="y")
        assert np.allclose(nd.data, 0.0)


# ---------------------------------------------------------------------------
# range_scale
# ---------------------------------------------------------------------------


class TestRangeScale:
    def test_basic(self, simple_2d):
        nd = range_scale(simple_2d, dim="y")
        col = simple_2d.data[:, 0]
        rng = np.max(col) - np.min(col)
        assert np.allclose(nd.data[:, 0], col / rng)

    def test_zero_range(self, simple_2d):
        const_data = np.ones((4, 6))
        const_ds = NDDataset(const_data, coordset=simple_2d.coordset)
        nd = range_scale(const_ds, dim="y")
        assert np.allclose(nd.data, 1.0)


# ---------------------------------------------------------------------------
# robust_scale
# ---------------------------------------------------------------------------


class TestRobustScale:
    def test_basic(self, simple_2d):
        nd = robust_scale(simple_2d, dim="y")
        col = simple_2d.data[:, 0]
        median = np.median(col)
        mad = np.median(np.abs(col - median)) * 1.4826
        expected = (col - median) / mad
        assert np.allclose(nd.data[:, 0], expected)

    def test_zero_mad(self, simple_2d):
        const_data = np.ones((4, 6))
        const_ds = NDDataset(const_data, coordset=simple_2d.coordset)
        nd = robust_scale(const_ds, dim="y")
        assert np.allclose(nd.data, 0.0)


# ---------------------------------------------------------------------------
# log_transform
# ---------------------------------------------------------------------------


class TestLogTransform:
    def test_log1p(self, simple_2d):
        nd = log_transform(simple_2d, method="log1p")
        expected = np.log1p(simple_2d.data)
        assert np.allclose(nd.data, expected)

    def test_log_positive(self, simple_2d):
        # Ensure all values are positive for plain log
        ds = simple_2d.copy()
        ds._data = ds.data + 10.0  # shift to positive
        nd = log_transform(ds, method="log")
        expected = np.log(ds.data)
        assert np.allclose(nd.data, expected)

    def test_log_with_offset(self, simple_2d):
        # Contains zeros, should auto-shift
        ds = simple_2d.copy()
        ds._data = np.zeros_like(ds.data)
        nd = log_transform(ds, method="log")
        assert np.allclose(nd.data, np.log(1e-10))

    def test_unknown_method(self, simple_2d):
        with pytest.raises(SpectroChemPyError, match="Unknown log_transform method"):
            log_transform(simple_2d, method="unknown")


# ---------------------------------------------------------------------------
# preprocessing transformers
# ---------------------------------------------------------------------------


class TestCenterTransformer:
    def test_fit_transform_agrees_with_center(self, simple_2d):
        expected = center(simple_2d, dim="y")
        scaler = CenterTransformer(dim="y")
        nd = scaler.fit_transform(simple_2d)
        assert np.allclose(nd.data, expected.data)
        assert "CenterTransformer applied on dimension y" in nd.history[-1]

    def test_transform_without_fit_raises(self, simple_2d):
        scaler = CenterTransformer(dim="y")
        with pytest.raises(SpectroChemPyError, match="not fitted yet"):
            scaler.transform(simple_2d)

    def test_inverse_transform_restores_data(self, simple_2d):
        scaler = CenterTransformer(dim="y")
        centered = scaler.fit_transform(simple_2d)
        restored = scaler.inverse_transform(centered)
        assert np.allclose(restored.data, simple_2d.data)
        assert "CenterTransformer inverse applied" in restored.history[-1]

    def test_train_test_reuse(self, simple_2d):
        train = simple_2d[:2]
        test = simple_2d[2:]
        scaler = CenterTransformer(dim="y")
        scaler.fit(train)
        test_centered = scaler.transform(test)
        # Mean must be the one learned from train, not test
        train_mean = np.mean(train.data, axis=0, keepdims=True)
        expected = test.data - train_mean
        assert np.allclose(test_centered.data, expected)

    def test_mask_and_metadata_preserved(self, simple_2d):
        ds = simple_2d.copy()
        ds._mask = np.zeros_like(ds.data, dtype=bool)
        ds._mask[0, 0] = True
        ds.meta.foo = "bar"
        scaler = CenterTransformer(dim="y")
        nd = scaler.fit_transform(ds)
        assert nd.meta.foo == "bar"
        assert nd.mask[0, 0]
        assert nd.coordset == ds.coordset


class TestAutoscaleTransformer:
    def test_fit_transform_agrees_with_autoscale(self, simple_2d):
        expected = autoscale(simple_2d, dim="y")
        scaler = AutoscaleTransformer(dim="y")
        nd = scaler.fit_transform(simple_2d)
        assert np.allclose(nd.data, expected.data)
        assert "AutoscaleTransformer applied on dimension y" in nd.history[-1]

    def test_transform_without_fit_raises(self, simple_2d):
        scaler = AutoscaleTransformer(dim="y")
        with pytest.raises(SpectroChemPyError, match="not fitted yet"):
            scaler.transform(simple_2d)

    def test_inverse_transform_restores_data(self, simple_2d):
        scaler = AutoscaleTransformer(dim="y")
        scaled = scaler.fit_transform(simple_2d)
        restored = scaler.inverse_transform(scaled)
        assert np.allclose(restored.data, simple_2d.data)
        assert "AutoscaleTransformer inverse applied" in restored.history[-1]

    def test_train_test_reuse(self, simple_2d):
        train = simple_2d[:2]
        test = simple_2d[2:]
        scaler = AutoscaleTransformer(dim="y")
        scaler.fit(train)
        test_scaled = scaler.transform(test)
        train_mean = np.mean(train.data, axis=0, keepdims=True)
        train_std = np.std(train.data, axis=0, keepdims=True)
        train_std_safe = np.where(train_std == 0, 1, train_std)
        expected = (test.data - train_mean) / train_std_safe
        assert np.allclose(test_scaled.data, expected)

    def test_zero_std_column(self, simple_2d):
        # Row 3 of simple_2d is constant; along dim=y its std is 0 for some columns
        scaler = AutoscaleTransformer(dim="y")
        nd = scaler.fit_transform(simple_2d)
        # Should not produce NaN
        assert not np.any(np.isnan(nd.data))
        restored = scaler.inverse_transform(nd)
        assert np.allclose(restored.data, simple_2d.data)


class TestSNVTransformer:
    def test_fit_transform_agrees_with_snv(self, simple_2d):
        expected = snv(simple_2d)
        scaler = SNVTransformer()
        nd = scaler.fit_transform(simple_2d)
        assert np.allclose(nd.data, expected.data)
        assert "SNVTransformer applied" in nd.history[-1]

    def test_transform_without_fit_raises(self, simple_2d):
        scaler = SNVTransformer()
        with pytest.raises(SpectroChemPyError, match="not fitted yet"):
            scaler.transform(simple_2d)

    def test_inverse_transform_restores_data(self, simple_2d):
        scaler = SNVTransformer()
        scaled = scaler.fit_transform(simple_2d)
        restored = scaler.inverse_transform(scaled)
        assert np.allclose(restored.data, simple_2d.data)
        assert "SNVTransformer inverse applied" in restored.history[-1]

    def test_dim_is_x(self, simple_2d):
        scaler = SNVTransformer()
        scaler.fit(simple_2d)
        assert scaler.dim == "x"

    def test_train_test_reuse(self, simple_2d):
        train = simple_2d[:2]
        test = simple_2d[2:]
        scaler = SNVTransformer()
        scaler.fit(train)
        test_snv = scaler.transform(test)
        # Stats computed per spectrum (dim=x)
        train_mean = np.mean(train.data, axis=1, keepdims=True)
        train_std = np.std(train.data, axis=1, keepdims=True)
        train_std_safe = np.where(train_std == 0, 1, train_std)
        expected = (test.data - train_mean) / train_std_safe
        assert np.allclose(test_snv.data, expected)
