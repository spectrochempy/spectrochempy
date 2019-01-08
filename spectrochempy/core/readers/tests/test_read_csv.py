# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import datadir
from spectrochempy.utils import show
from spectrochempy.utils.testing import assert_approx_equal
import pytest


def test_read_zip():

    with pytest.raises(NotImplementedError):
        A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip')

    A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip',
                           origin='omnic_export',
                           only=10)
    print(A)
    assert A.shape == (10, 3736)

    A.plot_stack()
    show()

def test_read_csv_tg():
    B = NDDataset.read_csv('agirdata/A350/TGA/tg.csv',
                           directory=datadir.path,
                           origin='tga')
    assert B.shape == (3446,)
    print(B)
    B.plot()
    show()


def test_read_csv_IR():
    B = NDDataset.read_csv('irdata/ir.csv', directory=datadir.path,
                           origin='ir', csv_delimiter=',')
    assert B.shape == (3736,)
    print(B)
    B.plot()
    show()
