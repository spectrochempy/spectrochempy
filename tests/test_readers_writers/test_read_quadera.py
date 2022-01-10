# -*- coding: utf-8 -*-
# flake8: noqa


from spectrochempy import NDDataset


# ..............................................................................
def test_read_quadera():
    # single file
    A = NDDataset.read_quadera("msdata/ion_currents.asc")
    assert str(A) == "NDDataset: [float64] A (shape: (y:16975, x:10))"
