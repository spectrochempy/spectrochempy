"""
NMR-specific 2D FFT encodings (STATES, TPPI, ECHO-ANTIECHO).

Each encoding handler receives quaternion data and returns quaternion data.
The quaternion → complex subspectra adaptation is delegated to the
hypercomplex representation layer (hypercomplex.py).

For 2D datasets the handler is called twice:
  1. F2 (direct dimension, original_axis=-1):
     Decompose quaternion using the encoding formula, FFT along axis=-1,
     rebuild quaternion from the FFT'd subspectra.
  2. F1 (indirect dimension, original_axis != -1):
     The rebuilt quaternion stores [Re(fr), Im(fr), Re(fi), Im(fi)].
     Extract complex directly (fr = RR + 1j*RI), FFT along axis=-1,
     rebuild quaternion from the result.
"""

from __future__ import annotations

import numpy as np


def _states_fft(data, tppi=False):
    """FFT transform according to STATES encoding."""
    from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components
    from spectrochempy_nmr.processing.hypercomplex import _prepare_states
    from spectrochempy_nmr.processing.hypercomplex import _rebuild_quaternion

    RR, RI, IR, II = _extract_quaternion_components(data)
    sr, si = _prepare_states(RR, RI, IR, II)

    if tppi:
        sr[..., 1::2] = -sr[..., 1::2]
        si[..., 1::2] = -si[..., 1::2]

    fr = np.fft.fftshift(np.fft.fft(sr), -1)
    fi = np.fft.fftshift(np.fft.fft(si), -1)

    return _rebuild_quaternion(fr, fi)


def _echoanti_fft(data):
    """FFT transform according to ECHO-ANTIECHO encoding."""
    from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components
    from spectrochempy_nmr.processing.hypercomplex import _prepare_echoanti
    from spectrochempy_nmr.processing.hypercomplex import _rebuild_quaternion

    RR, RI, IR, II = _extract_quaternion_components(data)
    c, s = _prepare_echoanti(RR, RI, IR, II)

    fc = np.fft.fftshift(np.fft.fft(c / 2.0), -1)
    fs = np.fft.fftshift(np.fft.fft(s / 2.0), -1)
    return _rebuild_quaternion(fc, fs)


def _tppi_fft(data):
    """FFT transform according to TPPI encoding."""
    from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components
    from spectrochempy_nmr.processing.hypercomplex import _prepare_tppi
    from spectrochempy_nmr.processing.hypercomplex import _rebuild_quaternion

    RR, RI, IR, II = _extract_quaternion_components(data)
    sx, sy = _prepare_tppi(RR, RI, IR, II)

    sx[..., 1::2] = -sx[..., 1::2]
    sy[..., 1::2] = -sy[..., 1::2]

    fx = np.fft.fftshift(np.fft.fft(sx), -1)
    fy = np.fft.fftshift(np.fft.fft(sy), -1)

    return _rebuild_quaternion(fx, fy)


def _qf_fft(data):
    """FFT transform according to QF encoding."""
    return np.fft.fftshift(np.fft.fft(np.conjugate(data)), -1)


def _fft_encoding_handler(data, encoding, **kwargs):
    """
    Dispatch NMR encoding-specific FFT transforms.

    Parameters
    ----------
    data : ndarray
        Input data (quaternion for first pass, quaternion rebuilt from
        previous pass for second pass).
    encoding : str
        The encoding identifier (e.g. "STATES", "DQD").
    **kwargs
        tppi : bool, optional
        original_axis : int
            The axis index *before* any swapdims.  When ``-1`` the handler
            is processing the direct (F2) dimension; otherwise the indirect
            (F1) dimension.  On the second pass the quaternion stores
            ``[Re(fr), Im(fr), Re(fi), Im(fi)]`` and must be decoded with
            direct extraction rather than the encoding-specific formula.
    """
    tppi = kwargs.get("tppi", False)
    original_axis = kwargs.get("original_axis", -1)

    # Second pass (indirect dimension): the quaternion was rebuilt from
    # [Re(fr), Im(fr), Re(fi), Im(fi)] by the first pass.  Extract the
    # complex subspectra directly and FFT along axis=-1 (which is F1 after
    # swapdims).
    if original_axis != -1:
        from spectrochempy_nmr.processing.hypercomplex import (
            _extract_quaternion_components,
        )
        from spectrochempy_nmr.processing.hypercomplex import _rebuild_quaternion

        RR, RI, IR, II = _extract_quaternion_components(data)
        fr = RR + 1j * RI
        fi = IR + 1j * II

        fr = np.fft.fftshift(np.fft.fft(fr), -1)
        fi = np.fft.fftshift(np.fft.fft(fi), -1)
        return _rebuild_quaternion(fr, fi)

    # First pass (direct dimension): use encoding-specific decomposition.
    if encoding in ("QSIM", "DQD"):
        return np.fft.fftshift(np.fft.fft(data), -1)
    if "QF" in encoding:
        return _qf_fft(data)
    if "QSEQ" in encoding:
        msg = "QSEQ NMR encoding is not yet implemented"
        raise NotImplementedError(msg)
    if "STATES" in encoding:
        return _states_fft(data, tppi=tppi)
    if "TPPI" in encoding and "STATES" not in encoding:
        return _tppi_fft(data)
    if "ECHO-ANTIECHO" in encoding:
        return _echoanti_fft(data)
    msg = f"NMR encoding {encoding!r} is not supported by the NMR plugin"
    raise NotImplementedError(msg)
