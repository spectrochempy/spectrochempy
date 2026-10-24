# ruff: noqa: S101
"""Characterize internal NMR linear prediction."""

from __future__ import annotations

import numpy as np
import pytest
from spectrochempy_nmr.processing.linear_prediction import _linear_predict_after


def test_forward_linear_prediction_extends_a_damped_complex_exponential():
    samples = np.arange(16)
    trace = 0.95**samples * np.exp(2j * np.pi * 0.125 * samples)

    actual = _linear_predict_after(trace, final_size=32, order=1)
    expected = 0.95 ** np.arange(32) * np.exp(2j * np.pi * 0.125 * np.arange(32))

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_forward_linear_prediction_preserves_real_trace_dtype():
    samples = np.arange(16)
    trace = 0.95**samples * np.cos(2 * np.pi * 0.125 * samples)

    actual = _linear_predict_after(trace, final_size=32, order=2)
    expected = 0.95 ** np.arange(32) * np.cos(2 * np.pi * 0.125 * np.arange(32))

    assert actual.dtype == trace.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_forward_linear_prediction_extends_each_trace_independently():
    samples = np.arange(12)
    data = np.stack((0.9**samples, 0.8**samples))

    actual = _linear_predict_after(data, final_size=20, order=1)

    np.testing.assert_allclose(actual[0], 0.9 ** np.arange(20), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], 0.8 ** np.arange(20), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
def test_forward_linear_prediction_preserves_low_precision_nmr_trace_dtypes(dtype):
    samples = np.arange(128)
    trace = 0.98**samples * np.exp(
        2j * np.pi * 0.12 * samples
    ) + 0.4 * 0.95**samples * np.exp(2j * np.pi * 0.16 * samples)
    noise = np.random.default_rng(12345).normal(scale=0.01, size=trace.shape)
    if dtype == np.float32:
        trace = (trace.real + noise).astype(dtype)
    else:
        trace = (trace + noise + 1j * noise[::-1]).astype(dtype)

    actual = _linear_predict_after(trace, final_size=256, order=32)

    assert actual.dtype == trace.dtype
    assert np.isfinite(actual).all()


@pytest.mark.parametrize(
    ("final_size", "order", "message"),
    ((3, 1, "final_size"), (5, 0, "order"), (5, 5, "order"), (5, 3, "half")),
)
def test_forward_linear_prediction_rejects_invalid_sizes(final_size, order, message):
    with pytest.raises(ValueError, match=message):
        _linear_predict_after(np.ones(4), final_size=final_size, order=order)


@pytest.mark.parametrize("invalid", (np.array(1.0), np.array([1.0, np.nan])))
def test_forward_linear_prediction_rejects_non_finite_or_scalar_data(invalid):
    with pytest.raises(ValueError):
        _linear_predict_after(invalid, final_size=4, order=1)


def test_forward_linear_prediction_rejects_integer_data():
    with pytest.raises(TypeError, match="dtype"):
        _linear_predict_after(np.ones(4, dtype=int), final_size=4, order=1)
