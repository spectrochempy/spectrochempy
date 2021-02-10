# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================


from spectrochempy import NDDataset


# ......................................................................................................................
def test_read_quadera():
    # single file
    A = NDDataset.read_quadera('msdata/ion_currents.asc')
    assert str(A) == 'NDDataset: [float64] A (shape: (y:4208, x:4))'
