# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841
"""Regression tests for the NMR digital filter removal algorithm."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from spectrochempy_nmr.extern.nmrglue import read_fid
from spectrochempy_nmr.readers.read_topspin import _remove_digital_filter

import spectrochempy as scp

_TOPSPIN_1D_FID = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_1d/1/fid"
_REFERENCE_FILE = Path(__file__).resolve().parent / "data/topspin_1d_rmdig_ref.npz"


def _make_dic(grpdly=None, dspfvs=20, decim=3328, td=None):
    acq = {"DECIM": decim, "DSPFVS": dspfvs, "TD": td if td is not None else 2048}
    if grpdly is not None:
        acq["GRPDLY"] = grpdly
    return {"acqus": acq}


def _get_function():
    from spectrochempy_nmr.readers.read_topspin import _remove_digital_filter

    return _remove_digital_filter


class TestRemoveDigitalFilter:
    """Tests for _remove_digital_filter."""

    def test_output_length_grpdly_integer(self):
        """GRPDLY=76 removes 78 points (floor(76)+2)."""
        fn = _get_function()
        n = 512
        data = np.exp(-np.arange(n) / 30.0).astype(np.complex128)
        dic = _make_dic(grpdly=76.0, td=2 * n)
        out = fn(dic, data.copy())
        assert len(out) == n - 78

    def test_output_length_grpdly_fractional(self):
        """GRPDLY=67.98 truncates to 67, removes 69 points."""
        fn = _get_function()
        n = 512
        data = np.exp(-np.arange(n) / 30.0).astype(np.complex128)
        dic = _make_dic(grpdly=67.9842681884766, td=2 * n)
        out = fn(dic, data.copy())
        assert len(out) == n - 69

    def test_dictionary_not_mutated(self):
        """The input dictionary is not modified."""
        fn = _get_function()
        n = 256
        data = np.exp(-np.arange(n) / 30.0).astype(np.complex128)
        dic = _make_dic(grpdly=76.0, td=2 * n)
        td_before = dic["acqus"]["TD"]
        fn(dic, data.copy())
        assert dic["acqus"]["TD"] == td_before

    def test_delay_correction(self):
        """The FID is delayed by floor(GRPDLY) points."""
        fn = _get_function()
        n = 256
        delay = 76
        # Build a reference (undelayed) signal
        t = np.arange(n)
        reference = np.exp(-t / 50.0).astype(np.complex128)
        # Build a delayed version: prepend `delay` zeros so the signal
        # starts at index `delay`, exactly as Bruker's digital filter does.
        fid_delayed = np.concatenate([np.zeros(delay), reference])
        dic = _make_dic(grpdly=float(delay), td=2 * len(fid_delayed))
        out = fn(dic, fid_delayed.copy())
        skip = int(np.floor(delay + 2.0))
        add = int(max(skip - 6, 0))
        # After correction, the output should match the undelayed reference
        # except for the tail-restoration region at the head.
        if add > 0:
            expected = reference[add : len(out)]
            actual = out[add:]
        else:
            expected = reference[: len(out)]
            actual = out
        assert np.allclose(actual, expected, atol=1e-10)

    def test_missing_acqus_raises(self):
        fn = _get_function()
        data = np.ones(64, dtype=np.complex128)
        with pytest.raises(ValueError, match="acqus"):
            fn({}, data)

    def test_missing_decim_raises(self):
        fn = _get_function()
        data = np.ones(64, dtype=np.complex128)
        dic = {"acqus": {"DSPFVS": 20}}
        with pytest.raises(ValueError, match="DECIM"):
            fn(dic, data)

    def test_missing_dspfvs_raises(self):
        fn = _get_function()
        data = np.ones(64, dtype=np.complex128)
        dic = {"acqus": {"DECIM": 2}}
        with pytest.raises(ValueError, match="DSPFVS"):
            fn(dic, data)

    def test_dspfvs_clamp(self):
        """DSPFVS < 10 is clamped to 10."""
        fn = _get_function()
        n = 256
        data = np.exp(-np.arange(n) / 30.0).astype(np.complex128)
        # DSPFVS=0, DECIM=2 -> clamped to DSPFVS=10, phase=44.75
        dic0 = _make_dic(grpdly=None, dspfvs=0, decim=2, td=2 * n)
        out0 = fn(dic0, data.copy())
        dic10 = _make_dic(grpdly=None, dspfvs=10, decim=2, td=2 * n)
        out10 = fn(dic10, data.copy())
        assert np.allclose(out0, out10)

    def test_dspfvs_14_no_phase(self):
        """DSPFVS >= 14 gives no phase correction."""
        fn = _get_function()
        n = 256
        data = np.exp(-np.arange(n) / 30.0).astype(np.complex128)
        dic = _make_dic(grpdly=None, dspfvs=14, decim=2, td=2 * n)
        out = fn(dic, data.copy())
        # phase=0, skip=2, so output length = n - 2
        assert len(out) == n - 2

    def test_known_good_reference_topspin_1d(self):
        """
        Compare `_remove_digital_filter` output to a static nmrglue-ng reference.

        Input: ``topspin_1d/1/fid`` (CC BY 4.0, Christian Fernandez, LCS).
        ``fid`` SHA-256
        ``a471ce0ec0c5f28b02c1ae05200f1b0300e29f042d36c3d5ff2d53c433219ed6``;
        ``acqus`` SHA-256
        ``00f86298b13a4f9a116feba96b9691fde6bf163ca4d1291a84d4ac8a82d12560``;
        ``DECIM=80``, ``DSPFVS=21``, ``GRPDLY=76``. The ``pdata/1/1r`` and
        ``1i`` pair is a processed spectrum (FFT, apodization, phase), not a
        filter-only reference, and is deliberately not used here.

        Expected: complex128, shape ``(12466,)``, produced once by nmrglue-ng
        ``e34ad7e90de215219087374009e6564fdb168220``
        (``nmrglue/fileio/bruker.py::remove_digital_filter``); raw-output
        SHA-256
        ``810a05b9471b6a47123b388883ff4137391ec079d797dd5a4e889196ed4d4a75``.
        The bundled ``read_fid`` feeds the exact reader path used by
        ``read_topspin``.

        Tolerance: ``atol=1e-8``/``rtol=1e-12`` leave a wide margin over the
        observed FFT last-bit scatter (max abs delta 4.66e-10, RMS 2.21e-11 on
        amplitudes up to ~2.1e6) while still pinning the numerical result.
        """
        if not _TOPSPIN_1D_FID.exists():
            pytest.skip(f"NMR test data not available: {_TOPSPIN_1D_FID}")

        dic, data = read_fid(_TOPSPIN_1D_FID.parent)
        out = _remove_digital_filter(dic, data)

        reference = np.load(_REFERENCE_FILE)["expected"]
        assert (
            hashlib.sha256(reference.tobytes()).hexdigest()
            == "810a05b9471b6a47123b388883ff4137391ec079d797dd5a4e889196ed4d4a75"
        )
        assert out.shape == reference.shape
        assert out.dtype == reference.dtype
        assert np.allclose(out, reference, rtol=1e-12, atol=1e-8)
