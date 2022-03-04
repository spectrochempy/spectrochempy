import spectrochempy as scp
import numpy as np


def test_baselinecorrector():

    X = scp.read_omnic("irdata/nh4y-activation.spg")
    X = X[0:10, 0:100]

    out = scp.BaselineCorrector(X)

    # no selection possible if we are not in a notebook
    assert out.corrected.shape == (10, 100)
    assert len(out._fig.axes[0].lines) == 20, "original + baselines"
    assert len(out._fig.axes[1].lines) == 10, "corrected"
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == X.x.data)

    # try higher polyorder
    out._orderslider.value = 3
    out.process_clicked()
    assert out.corrected.shape == (10, 100)
    assert len(out._fig.axes[0].lines) == 20, "original + baselines"
    assert len(out._fig.axes[1].lines) == 10, "corrected"

    # try multivariate
    out._methodselector.value = "multivariate"

    # try multivariate, with 2 pcs
    out._npcslider.value = 2

    out.process_clicked()
