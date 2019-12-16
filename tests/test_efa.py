# -*- coding: utf-8 -*-
#
# ======================================================================================================================

# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from spectrochempy.core.analysis.efa import EFA
from spectrochempy.utils import MASKED, show
from spectrochempy.core import info_


def test_EFA(IR_dataset_2D):

    ds = IR_dataset_2D.copy()

    info_(ds)

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

    show()




# ======================================================================================================================
if __name__ == '__main__':
    pass
