import spectrochempy as scp
import numpy as np


def test_baselinecorrector():

    out = scp.BaselineCorrector()
    assert out._fig is None, "No plot"
    assert not hasattr(out, "original")
    assert out.corrected.is_empty
    # check uploader (see https://github.com/jupyter-widgets/ipywidgets/blob/master
    # /python/ipywidgets/ipywidgets/widgets/tests/test_widget_upload.py)
    datadir = scp.preferences.datadir
    with open(datadir / "irdata/nh4y-activation.spg", "rb") as fil:
        content = fil.read()

    CONTENT = {"nh4y-activation.spg": {"content": memoryview(content)}}
    out._uploader.set_trait("value", CONTENT)
    out.process_clicked()
    assert out.original.name == "nh4y-activation"
    assert out.original.shape == (55, 5549)

    CONTENT2 = {}
    for i in range(4):
        with open(datadir / f"irdata/OPUS/test.000{i}", "rb") as fil:
            content = fil.read()
            CONTENT2.update({f"test.000{i}": {"content": memoryview(content)}})
    out = scp.BaselineCorrector()
    out._uploader.set_trait("value", CONTENT2)
    out.process_clicked()
    assert out.original.name == "test.0003"
    assert out.original.shape == (4, 2567)

    out = scp.BaselineCorrector()
    CONTENT3 = CONTENT
    CONTENT3.update(CONTENT2)
    out._uploader.set_trait("value", CONTENT3)
    out.process_clicked()
    assert not hasattr(out, "original")

    X = scp.read_omnic("irdata/nh4y-activation.spg")
    _X = X[0:10, 0:100]
    out = scp.BaselineCorrector(_X)
    assert out.corrected.shape == (10, 100)
    assert len(out._fig.axes[0].lines) == 20, "original + baselines"
    assert len(out._fig.axes[1].lines) == 10, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == _X.x.data)

    # try higher polyorder
    out._orderslider.value = 3
    out.process_clicked()
    assert out.corrected.shape == (10, 100)
    assert len(out._fig.axes[0].lines) == 20, "original + baselines"
    assert len(out._fig.axes[1].lines) == 10, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == _X.x.data)

    # try multivariate
    out._methodselector.value = "multivariate"

    # try multivariate, with 2 pcs
    out._npcslider.value = 2

    out = scp.BaselineCorrector(X)
    out._x_limits_control.value = "[5000.56 : 649.9 : 1]"
    out._y_limits_control.value = "[0:55:2]"
    out.process_clicked()
    assert len(out._fig.axes[0].lines) == 56, "original + baselines"
    assert len(out._fig.axes[1].lines) == 28, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == X.x[5000.56:649.9].data)

    initial_ranges = (
        [5900.0, 5400.0],
        [4000.0, 4500.0],
        4550.0,
        [2100.0, 2000.0],
        [1550.0, 1555.0],
        [1250.0, 1300.0],
        [800.0, 850.0],
    )
    out = scp.BaselineCorrector(X, initial_ranges=initial_ranges)
    out._x_limits_control.value = "[5400.56 : 800.9 : 1]"
    out.process_clicked()
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == X.x[5400.56:800.9].data)

    out2 = scp.BaselineCorrector(X)
    out2._x_limits_control.value = "[5400.56 : 800.9 : 1]"
    out2._ranges_control.value = """
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
    out2.process_clicked()
    assert np.all(out2._fig.axes[1].lines[0].get_xdata() == X.x[5400.56:800.9].data)
    assert np.all(
        out._fig.axes[1].lines[0].get_xdata() == out2._fig.axes[1].lines[0].get_xdata()
    )
