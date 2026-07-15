"""
Hypercomplex representation adaptation layer for NMR 2D FFT.

This module separates the quaternion → complex subspectra adaptation from
the FFT transform. Each encoding scheme defines its own projection of the
quaternion components (RR, RI, IR, II) onto two complex subspectra.

The result is two complex arrays (one pair per subspectrum) that can be
processed independently by standard complex FFT.

Architecture:
    quaternion (from reader / encoding)
        │
        ▼
    _extract_quaternion_components(data) → (RR, RI, IR, II)
        │
        ▼
    _prepare_states / _prepare_tppi / _prepare_echoanti
        │
        ▼
    two complex subspectra
        │
        ▼
    FFT (standard complex)
"""

from __future__ import annotations


def _extract_quaternion_components(data):
    """
    Extract the four real components from a quaternion array.

    Parameters
    ----------
    data : quaternion array
        Quaternion data as returned by the reader (e.g., TopSpin).

    Returns
    -------
    RR, RI, IR, II : ndarray
        The four real component arrays.

    Raises
    ------
    ImportError
        If spectrochempy-hypercomplex is not installed.

    Notes
    -----
    The numpy-quaternion convention is:
        as_float_array(q) → [..., 4] with order [w, x, y, z]
        w = RR, x = RI, y = IR, z = II
    """
    try:
        from spectrochempy_hypercomplex import as_float_array  # noqa: PLC0415
    except ModuleNotFoundError:
        msg = (
            "2D hypercomplex NMR processing requires the spectrochempy-hypercomplex "
            "plugin. Install it with: pip install spectrochempy-hypercomplex"
        )
        raise ImportError(msg) from None

    fa = as_float_array(data)
    RR = fa[..., 0]
    RI = fa[..., 1]
    IR = fa[..., 2]
    II = fa[..., 3]
    return RR, RI, IR, II


def _prepare_states(RR, RI, IR, II):
    """
    Prepare complex subspectra for STATES encoding.

    Projection:
        sr = (RR - 1j * RI) / 2
        si = (IR - 1j * II) / 2

    Parameters
    ----------
    RR, RI, IR, II : ndarray
        Quaternion components.

    Returns
    -------
    sr, si : ndarray
        Two complex subspectra.
    """
    sr = (RR - 1j * RI) / 2.0
    si = (IR - 1j * II) / 2.0
    return sr, si


def _prepare_tppi(RR, RI, IR, II):
    """
    Prepare complex subspectra for TPPI encoding.

    Projection:
        sx = RR + 1j * IR
        sy = RI + 1j * II

    Parameters
    ----------
    RR, RI, IR, II : ndarray
        Quaternion components.

    Returns
    -------
    sx, sy : ndarray
        Two complex subspectra.
    """
    sx = RR + 1j * IR
    sy = RI + 1j * II
    return sx, sy


def _prepare_echoanti(RR, RI, IR, II):
    """
    Prepare complex subspectra for Echo-Antiecho encoding.

    Projection:
        c = (RR + IR) + 1j * (RR - IR)
        s = (RI + II) - 1j * (RI - II)

    Parameters
    ----------
    RR, RI, IR, II : ndarray
        Quaternion components.

    Returns
    -------
    c, s : ndarray
        Two complex subspectra.
    """
    c = (RR + IR) + 1j * (RR - IR)
    s = (RI + II) - 1j * (RI - II)
    return c, s


def _rebuild_quaternion(fr, fi):
    """
    Rebuild quaternion from two complex subspectra.

    Parameters
    ----------
    fr, fi : ndarray
        Two complex arrays.

    Returns
    -------
    quaternion array
        Quaternion with components [fr.real, fr.imag, fi.real, fi.imag].

    Raises
    ------
    ImportError
        If spectrochempy-hypercomplex is not installed.
    """
    try:
        from spectrochempy_hypercomplex import as_quaternion  # noqa: PLC0415
    except ModuleNotFoundError:
        msg = (
            "2D hypercomplex NMR processing requires the spectrochempy-hypercomplex "
            "plugin. Install it with: pip install spectrochempy-hypercomplex"
        )
        raise ImportError(msg) from None

    return as_quaternion(fr, fi)
