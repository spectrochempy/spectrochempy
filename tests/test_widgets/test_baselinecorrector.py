# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import numpy as np
import pytest

import spectrochempy
import spectrochempy as scp


DATADIR = scp.preferences.datadir
SPG_FILE = DATADIR / "irdata/nh4y-activation.spg"
UNREADABLE = DATADIR / "irdata/omnic_series/GC_Demo.srs"


@pytest.fixture(scope="module")
def X():
    X = scp.read_omnic(SPG_FILE)
    X.y -= X.y[0]
    return X


def test_baselinecorrector_load_clicked(X, monkeypatch):
    def open_ok(*args, **kwargs):
        # mock opening a dialog
        return SPG_FILE

    def open_cancel(*args, **kwargs):
        # mock dialog canceled
        return None

    def open_wrong(*args, **kwargs):
        # mock opening a dialog
        return UNREADABLE

    out = scp.BaselineCorrector()
    assert out.corrected.is_empty
    # save
    # write without parameters and dialog cancel
    monkeypatch.setenv(
        "KEEP_DIALOGS", "True"
    )  # we ask to display dialogs as we will mock them.

    monkeypatch.setattr(spectrochempy.core, "open_dialog", open_cancel)
    assert out._load_clicked() is None

    monkeypatch.setattr(spectrochempy.core, "open_dialog", open_ok)
    out._load_clicked()
    assert out.original.name == "nh4y-activation"

    out = scp.BaselineCorrector()
    monkeypatch.setattr(spectrochempy.core, "open_dialog", open_wrong)
    out._load_clicked()
    assert out.original.is_empty


def test_baselinecorrector_slicing(X):

    out = scp.BaselineCorrector(X)

    assert out.corrected.shape == (55, 5549)
    assert len(out._fig.axes[0].lines) == 110, "original + baselines"
    assert len(out._fig.axes[1].lines) == 55, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == X.x.data)
    assert np.all(out._fig.axes[0].lines[10].get_ydata() == X.data[10])

    # slicing
    out._x_limits_control.value = "[5000.56 : 649.9 : 1]"
    out._y_limits_control.value = "[0:55:2]"
    out._process_clicked()
    assert len(out._fig.axes[0].lines) == 56, "original + baselines"
    assert len(out._fig.axes[1].lines) == 28, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == X.x[5000.56:649.9].data)
    assert np.all(out._fig.axes[0].lines[1].get_ydata() == X[2, 5000.56:649.9].data)

    out._x_limits_control.value = "[5400.56 : 800.9 : 1]"
    out._ranges_control.value = """
    (
    [5900.0, 5400.0],
        [4000.0, 4500.0],
       4550.0,
    [2100.0, 2000.0],
    [1550.0, 1555.0],
    [1250.0, 1300.0],
    [800.0, 850.0],
    )
    """
    out._process_clicked()
    assert out.corrected.shape == (28, 4771)
    initial_ranges = (
        [5900.0, 5400.0],
        [4000.0, 4500.0],
        4550.0,
        [2100.0, 2000.0],
        [1550.0, 1555.0],
        [1250.0, 1300.0],
        [800.0, 850.0],
    )
    out2 = scp.BaselineCorrector(X, initial_ranges=initial_ranges)
    out2._x_limits_control.value = "[5400.56 : 800.9 : 1]"
    out2._process_clicked()
    assert np.all(out2._fig.axes[1].lines[0].get_xdata() == X.x[5400.56:800.9].data)

    assert np.all(
        out._fig.axes[1].lines[0].get_xdata() == out2._fig.axes[1].lines[0].get_xdata()
    )

    # slicing limits out of coord range
    out = scp.BaselineCorrector(X[::50, :])
    out._x_limits_control.value = "[4000.0 : 2000.0 : 1]"
    out._ranges_control.value = """
    (
    [5900.0, 5400.0],
    [800.0, 850.0],
    )
    """
    out._process_clicked()
    assert out.corrected.shape == (2, 2075)

    out._ranges_control.value = """
    (
    5900.,
    [800.0, 850.0],
    )
    """
    out._process_clicked()
    assert out.corrected.shape == (2, 2075)

    out._ranges_control.value = """
    (
    [5900.0, 5400.0],
    850.0
    )
    """
    out._process_clicked()
    assert out.corrected.shape == (2, 2075)

    # other slicing format
    out = scp.BaselineCorrector(X)
    out._y_limits_control.value = "[0.0:3000.0:1]"  # y location
    out._process_clicked()
    assert out.corrected.shape == (6, 5549)

    # missing parts in slice
    out._y_limits_control.value = "[:3000.0]"  # y location
    out._process_clicked()
    assert out.corrected.shape == (6, 5549)

    out._y_limits_control.value = "::2"  # y location
    out._process_clicked()
    assert out.corrected.shape == (28, 5549)

    out._y_limits_control.value = "0:10:2"  # y location
    out._process_clicked()
    assert out.corrected.shape == (5, 5549)

    out._x_limits_control.value = "::100"  # y location
    out._process_clicked()
    assert out.corrected.shape == (5, 56)

    out._x_limits_control.value = "::100:"  # y location
    with pytest.raises(ValueError):
        out._process_clicked()


def test_baselinecorrector_not_a_NDDataset(X):
    with pytest.raises(ValueError):
        scp.BaselineCorrector(X.x)


def test_baselinecorrector_parameters(X):
    _X = X[0:10, 0:100]
    out = scp.BaselineCorrector(_X)
    # sequential
    assert out._method_selector in out._method_control.children
    assert out._npc_slider not in out._method_control.children

    out._interpolation_selector.value = "pchip"
    out._process_clicked()

    # try higher polyorder
    out._order_slider.value = 3
    out._interpolation_selector.value = "polynomial"
    out._process_clicked()

    assert out.corrected.shape == (10, 100)
    assert len(out._fig.axes[0].lines) == 20, "original + baselines"
    assert len(out._fig.axes[1].lines) == 10, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == _X.x.data)

    # try multivariate
    out._method_selector.value = "multivariate"
    assert out._method_selector in out._method_control.children
    assert out._npc_slider in out._method_control.children

    # try multivariate, with 2 pcs
    out._npc_slider.value = 2
    out._process_clicked()

    out._method_selector.value = "sequential"
    assert out._method_selector in out._method_control.children
    assert out._npc_slider not in out._method_control.children


def test_baselinecorrector_save_clicked(X, monkeypatch):
    def dialog_cancel(*args, **kwargs):
        # mock a dialog cancel action
        return None

    def dialog_save(*args, **kwargs):
        # mock a dialog to save
        return "spec.scp"

    out = scp.BaselineCorrector(X)
    # save
    # write without parameters and dialog cancel
    monkeypatch.setenv(
        "KEEP_DIALOGS", "True"
    )  # we ask to display dialogs as we will mock them.

    monkeypatch.setattr(spectrochempy.core, "save_dialog", dialog_cancel)
    assert out._save_clicked() is None

    monkeypatch.setattr(spectrochempy.core, "save_dialog", dialog_save)
    filename = out._save_clicked()
    assert filename == scp.pathclean("spec.scp")  # <-
    assert filename.exists()
    if filename.exists:
        filename.unlink()
