"""NMR-specific 2D FFT encodings (STATES, TPPI, ECHO-ANTIECHO)."""

from __future__ import annotations

import numpy as np


def _states_fft(data, tppi=False):
    """FFT transform according to STATES encoding."""
    from spectrochempy_hypercomplex._quaternion import as_float_array  # noqa: PLC0415
    from spectrochempy_hypercomplex._quaternion import as_quaternion  # noqa: PLC0415

    # warning: at this point, data must have been swapped so the last dimension is the one used for FFT
    wt, yt, xt, zt = as_float_array(
        data
    ).T  # x and y are exchanged due to swapping of dims
    w, y, x, z = wt.T, yt.T, xt.T, zt.T

    sr = (w - 1j * y) / 2.0
    si = (x - 1j * z) / 2.0

    if tppi:
        sr[..., 1::2] = -sr[..., 1::2]
        si[..., 1::2] = -si[..., 1::2]

    fr = np.fft.fftshift(np.fft.fft(sr), -1)
    fi = np.fft.fftshift(np.fft.fft(si), -1)

    # rebuild the quaternion
    return as_quaternion(fr, fi)


def _echoanti_fft(data):
    """FFT transform according to ECHO-ANTIECHO encoding."""
    from spectrochempy_hypercomplex._quaternion import as_float_array  # noqa: PLC0415
    from spectrochempy_hypercomplex._quaternion import as_quaternion  # noqa: PLC0415

    wt, yt, xt, zt = as_float_array(data).T
    w, y, x, z = wt.T, xt.T, yt.T, zt.T

    c = (w + y) + 1j * (w - y)
    s = (x + z) - 1j * (x - z)
    fc = np.fft.fftshift(np.fft.fft(c / 2.0), -1)
    fs = np.fft.fftshift(np.fft.fft(s / 2.0), -1)
    return as_quaternion(fc, fs)


def _tppi_fft(data):
    """FFT transform according to TPPI encoding."""
    from spectrochempy_hypercomplex._quaternion import as_float_array  # noqa: PLC0415
    from spectrochempy_hypercomplex._quaternion import as_quaternion  # noqa: PLC0415

    wt, yt, xt, zt = as_float_array(data).T
    w, y, x, z = wt.T, xt.T, yt.T, zt.T

    sx = w + 1j * y
    sy = x + 1j * z

    sx[..., 1::2] = -sx[..., 1::2]
    sy[..., 1::2] = -sy[..., 1::2]

    fx = np.fft.fftshift(np.fft.fft(sx), -1)  # reverse
    fy = np.fft.fftshift(np.fft.fft(sy), -1)

    # rebuild the quaternion
    return as_quaternion(fx, fy)


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
