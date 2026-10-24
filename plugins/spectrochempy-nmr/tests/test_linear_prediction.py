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


@pytest.mark.parametrize(
    ("final_size", "order", "message"),
    ((3, 1, "final_size"), (5, 0, "order"), (5, 5, "order")),
)
def test_forward_linear_prediction_rejects_invalid_sizes(final_size, order, message):
    with pytest.raises(ValueError, match=message):
        _linear_predict_after(np.ones(4), final_size=final_size, order=order)
