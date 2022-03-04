import spectrochempy as scp
import numpy as np


def test_baselinecorrector():

    X = scp.read_omnic("irdata/nh4y-activation.spg")
    X = X[0:10, 0:100]
    initial_ranges = (
        [5900.0, 5400.0],
        [4000.0, 4500.0],
    )
    out = scp.BaselineCorrector(X, initial_ranges=initial_ranges)
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
    assert np.all(out._fig.axes[1].lines[0].get_xdata() == X.x.data)

    # try multivariate
    out._methodselector.value = "multivariate"

    # try multivariate, with 2 pcs
    out._npcslider.value = 2

    out.process_clicked()
