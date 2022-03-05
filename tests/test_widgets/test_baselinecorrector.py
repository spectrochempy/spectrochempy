import numpy as np
import pytest

import spectrochempy
import spectrochempy as scp


@pytest.fixture(scope="module")
def X():
    X = scp.read_omnic("irdata/nh4y-activation.spg")
    X.y -= X.y[0]
    return X


def test_baselinecorrector_uploader():
    # check uploader

    # single file
    out = scp.BaselineCorrector()
    assert out._fig is None, "No plot"
    assert not hasattr(out, "original")
    assert out.corrected.is_empty
    datadir = scp.preferences.datadir
    with open(datadir / "irdata/nh4y-activation.spg", "rb") as fil:
        content = fil.read()
    CONTENT = {"nh4y-activation.spg": {"content": memoryview(content)}}
    out._uploader.set_trait("value", CONTENT)
    out.process_clicked()
    assert out.original.name == "nh4y-activation"
    assert out.original.shape == (55, 5549)

    # multiple files that can be merged
    out = scp.BaselineCorrector()
    CONTENT2 = {}
    for i in range(4):
        with open(datadir / f"irdata/OPUS/test.000{i}", "rb") as fil:
            content = fil.read()
            CONTENT2.update({f"test.000{i}": {"content": memoryview(content)}})
    out._uploader.set_trait("value", CONTENT2)
    out.process_clicked()
    assert out.original.name == "test.0003"
    assert out.original.shape == (4, 2567)

    # incompatibles files
    out = scp.BaselineCorrector()
    CONTENT3 = CONTENT
    CONTENT3.update(CONTENT2)
    out._uploader.set_trait("value", CONTENT3)
    out.process_clicked()
    assert not hasattr(out, "original")


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
    out.process_clicked()
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
    out.process_clicked()
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
    out2.process_clicked()
    assert np.all(out2._fig.axes[1].lines[0].get_xdata() == X.x[5400.56:800.9].data)

    assert np.all(
        out._fig.axes[1].lines[0].get_xdata() == out2._fig.axes[1].lines[0].get_xdata()
    )

    # reset slicing
    out._x_limits_control.value = "[5999.56 : 649.9 : 1]"
    out._y_limits_control.value = "[0:55:1]"
    out.process_clicked()
    assert out.corrected.shape == (55, 5549)

    # other slicing format
    out._y_limits_control.value = "[0.0:3000.0:1]"  # y location
    out.process_clicked()
    assert out.corrected.shape == (6, 5549)

    # missing parts in slice
    out._y_limits_control.value = "[:3000.0]"  # y location
    out.process_clicked()
    assert out.corrected.shape == (6, 5549)

    out._y_limits_control.value = "::2"  # y location
    out.process_clicked()
    assert out.corrected.shape == (28, 5549)

    out._y_limits_control.value = "0:10:2"  # y location
    out.process_clicked()
    assert out.corrected.shape == (5, 5549)

    out._x_limits_control.value = "::100"  # y location
    out.process_clicked()
    assert out.corrected.shape == (5, 56)


def test_baselinecorrector_not_a_NDDataset(X):

    with pytest.raises(ValueError):
        scp.BaselineCorrector(X.x)


def test_baselinecorrector_parameters(X):

    _X = X[0:10, 0:100]
    out = scp.BaselineCorrector(_X)
    # sequential
    assert out._methodselector in out._method_control.children
    assert out._npcslider not in out._method_control.children

    out._interpolationselector.value = "pchip"
    out.process_clicked()

    # try higher polyorder
    out._orderslider.value = 3
    out._interpolationselector.value = "polynomial"
    out.process_clicked()

    assert out.corrected.shape == (10, 100)
    assert len(out._fig.axes[0].lines) == 20, "original + baselines"
    assert len(out._fig.axes[1].lines) == 10, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == _X.x.data)

    # try multivariate
    out._methodselector.value = "multivariate"
    assert out._methodselector in out._method_control.children
    assert out._npcslider in out._method_control.children

    # try multivariate, with 2 pcs
    out._npcslider.value = 2
    out.process_clicked()

    out._methodselector.value = "sequential"
    assert out._methodselector in out._method_control.children
    assert out._npcslider not in out._method_control.children


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
    assert out.save_clicked() is None

    monkeypatch.setattr(spectrochempy.core, "save_dialog", dialog_save)
    filename = out.save_clicked()
    assert filename == scp.pathclean("spec.scp")  # <-
    assert filename.exists()
    if filename.exists:
        filename.unlink()
