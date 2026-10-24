"""Internal linear-prediction helpers for NMR time-domain traces."""

from __future__ import annotations

import numpy as np


def _linear_predict_after(data, final_size, order):
    """
    Extend time-domain traces along their last axis with forward LP.

    Each trace is fitted independently by a least-squares autoregressive model
    of ``order`` points. Roots outside the unit circle are projected onto it to
    prevent an increasing extrapolation.
    """
    values = np.asarray(data)
    if values.ndim == 0:
        msg = "data must have at least one dimension"
        raise ValueError(msg)
    if values.dtype.kind not in "fc":
        msg = "data must have a floating-point or complex dtype"
        raise TypeError(msg)
    if not np.isfinite(values).all():
        msg = "data must contain only finite values"
        raise ValueError(msg)
    initial_size = values.shape[-1]
    if final_size < initial_size:
        msg = "final_size must be at least the input size"
        raise ValueError(msg)
    if not 0 < order <= initial_size // 2:
        msg = "order must be positive and at most half the input size"
        raise ValueError(msg)

    output = np.empty((*values.shape[:-1], final_size), dtype=values.dtype)
    output[..., :initial_size] = values
    if final_size == initial_size:
        return output

    for trace in output.reshape(-1, final_size):
        coefficients = _forward_coefficients(trace[:initial_size], order)
        for index in range(initial_size, final_size):
            trace[index] = np.dot(trace[index - order : index], coefficients)
    return output


def _forward_coefficients(trace, order):
    """Fit and stabilize the forward autoregressive coefficients for one trace."""
    rows = trace.size - order
    matrix = np.empty((rows, order), dtype=trace.dtype)
    for index in range(rows):
        matrix[index] = trace[index : index + order]
    coefficients = np.linalg.lstsq(matrix, trace[order:], rcond=None)[0]

    polynomial = np.empty(order + 1, dtype=coefficients.dtype)
    polynomial[0] = 1.0
    polynomial[1:] = -coefficients[::-1]
    roots = np.roots(polynomial)
    roots[np.abs(roots) > 1.0] /= np.abs(roots[np.abs(roots) > 1.0])
    coefficients = -np.poly(roots)[:0:-1]
    if np.isrealobj(trace):
        coefficients = np.real_if_close(coefficients)
    return coefficients
