from spectrochempy import *


def test_issue_15():
    X = read_omnic('irdata/nh4y-activation.spg')
    mypca = PCA(X)
    xhat = mypca.reconstruct(n_pc=3)