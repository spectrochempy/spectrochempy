# ======================================================================================
# Copyright (c) 2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Displaying objects must not alter NumPy's global print options (#1974)."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture
def user_printoptions():
    saved = np.get_printoptions()
    np.set_printoptions(threshold=1000)
    yield
    np.set_printoptions(**saved)


def test_display_keeps_global_numpy_printoptions(user_printoptions):
    ds = scp.NDDataset(np.random.default_rng(0).random((6, 20)))
    ds.y = scp.Coord(labels=list("abcdef"))
    before = np.get_printoptions()
    for obj in (ds, ds.x, ds.y, ds.coordset):
        str(obj)
        repr(obj)
        if hasattr(obj, "_repr_html_"):
            obj._repr_html_()
    assert np.get_printoptions() == before
    assert np.get_printoptions()["threshold"] == 1000
    assert str(np.arange(12.0)) == "[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]"


def test_import_and_display_independent_of_user_printoptions():
    # User options are set *before* importing/displaying: the dataset text must be
    # the pinned SpectroChemPy layout and the user options must survive.
    code = textwrap.dedent(
        """
        import numpy as np
        np.set_printoptions(threshold=10, precision=1, edgeitems=1)
        before = np.get_printoptions()
        import spectrochempy as scp
        assert np.get_printoptions() == before, "import changed print options"
        ds = scp.NDDataset(np.arange(40.0).reshape(4, 10), name="x", title="t")
        print(repr(ds._str_value()))
        str(ds); repr(ds); ds._repr_html_()
        assert np.get_printoptions() == before, "display changed print options"
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == repr(
        "          DATA \n        title: t\n       values: ... \n\x00"
        "         [[       0        1 ...        8        9]\n"
        "          [      10       11 ...       18       19]\n"
        "          [      20       21 ...       28       29]\n"
        "          [      30       31 ...       38       39]]\x00"
    )


def _entry(ds):
    return ds.history[-1].split("> ", 1)[1]


def test_binary_operation_history_text_is_unchanged():
    ds = scp.NDDataset(np.ones((30, 30)), name="b")
    row = "[     0.5      0.5 ...      0.5      0.5]"
    block = "[" + "\n ".join([row, row, "...", row, row]) + "]"
    assert _entry(ds + np.full((30, 30), 0.5)) == (
        f"Binary operation add with `{block}` has been performed"
    )
    assert _entry(ds - np.arange(30.0)) == (
        "Subtracted `[       0        1 ...       28       29]` from `b`"
    )
    assert _entry(ds * scp.Quantity(np.arange(30.0), "m")) == (
        "Binary operation mul with `[0.0 1.0 ... 28.0 29.0] m` has been performed"
    )
    ip = ds.copy()
    ip *= np.arange(30.0)
    assert _entry(ip) == (
        "Inplace binary op: imul  with `[       0        1 ...       28       29]` "
    )


@pytest.mark.parametrize("mapping", ["parameters", "diagnostics"])
def test_analysis_result_array_display_preserves_format_and_options(mapping):
    from spectrochempy.analysis._base._result import AnalysisResult

    expected = "[       0   0.3333 ...    2.667        3]"
    result = AnalysisResult(
        estimator="example", **{mapping: {"values": np.arange(10.0) / 3}}
    )
    with np.printoptions(precision=1, threshold=1000, edgeitems=1):
        before = np.get_printoptions()
        assert result._repr_sections()[-1].items[0].value == expected
        assert expected in result._repr_html_()
        if mapping == "parameters":
            assert expected in repr(result)
        assert np.get_printoptions() == before


@pytest.mark.parametrize("mapping", ["parameters", "diagnostics"])
def test_analysis_result_display_restores_options_after_exception(mapping):
    from spectrochempy.analysis._base._result import AnalysisResult

    class BrokenValue:
        def __str__(self):
            raise ValueError("cannot display")

    result = AnalysisResult(estimator="example", **{mapping: {"bad": BrokenValue()}})
    with np.printoptions(precision=1, threshold=1000, edgeitems=1):
        before = np.get_printoptions()
        with pytest.raises(ValueError, match="cannot display"):
            result._repr_html_()
        assert np.get_printoptions() == before
        if mapping == "parameters":
            with pytest.raises(ValueError, match="cannot display"):
                repr(result)
            assert np.get_printoptions() == before
