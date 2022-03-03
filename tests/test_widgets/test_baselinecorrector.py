import spectrochempy as scp


def test_baselinecorrector():

    X = scp.read_omnic("irdata/nh4y-activation.spg")
    out = scp.BaselineCorrector(X[0:10, 0:100])
    # no selection possible if we are not in a notebook
    assert out.corrected.shape == (10, 99)

    # now change widgets settings to increase coverage,
    # but not able to trigger "process": out._process_clicked() not recognized)

    # try higher polyorder
    out._orderslider.value = 3

    # try multivariate
    out._methodselector.value = "multivariate"

    # try multivariate, with 2 pcs
    out._npcslider.value = 2
