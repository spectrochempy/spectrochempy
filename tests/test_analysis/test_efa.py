# -*- coding: utf-8 -*-
# flake8: noqa


from spectrochempy.core.analysis.efa import EFA
from spectrochempy.utils import MASKED


def test_EFA(IR_dataset_2D):
    ds = IR_dataset_2D.copy()

    # columns masking
    ds[:, 1230.0:920.0] = MASKED  # do not forget to use float in slicing
    ds[:, 5900.0:5890.0] = MASKED

    # difference spectra
    ds -= ds[-1]

    # column masking for bad columns
    ds[10:12] = MASKED

    efa = EFA(ds)

    n_pc = 4
    c = efa.get_conc(n_pc)
    c.T.plot()

    # show()
