"""
NMR-specific 2D FFT encodings (STATES, TPPI, ECHO-ANTIECHO).

Each encoding handler receives quaternion data and returns quaternion data.
The quaternion → complex subspectra adaptation is delegated to the
hypercomplex representation layer (hypercomplex.py).

The handler only performs:
    1. Encoding-specific processing (sign alternation)
    2. Standard complex FFT
    3. Quaternion reconstruction
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
    """Dispatch NMR encoding-specific FFT transforms."""
    tppi = kwargs.get("tppi", False)
    if encoding in ("QSIM", "DQD"):
        # Standard complex FFT – core already handles this, but we keep it
        # here for explicitness when a plugin encoding handler is queried.
        return np.fft.fftshift(np.fft.fft(data), -1)
    if "QF" in encoding:
        return _qf_fft(data)
    if "QSEQ" in encoding:
        raise NotImplementedError("QSEQ NMR encoding is not yet implemented")
    if "STATES" in encoding:
        return _states_fft(data, tppi=tppi)
    if "TPPI" in encoding and "STATES" not in encoding:
        return _tppi_fft(data)
    if "ECHO-ANTIECHO" in encoding:
        return _echoanti_fft(data)
    msg = f"NMR encoding {encoding!r} is not supported by the NMR plugin"
    raise NotImplementedError(msg)
