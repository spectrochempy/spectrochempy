# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841
"""Regression tests for the NMR digital filter removal algorithm."""

import numpy as np
import pytest


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
