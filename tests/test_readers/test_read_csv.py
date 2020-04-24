# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================


from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import general_preferences as prefs
from spectrochempy.utils import show
from spectrochempy.utils.testing import assert_approx_equal
import pytest

def test_read_zunimplemented():

    with pytest.raises(NotImplementedError):
        A = NDDataset.read_zip('agirdata/P350/FTIR/FTIR.zip')

def test_read_zip():

    #with pytest.raises(NotImplementedError):
    #    A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip')

    A = NDDataset.read_zip('agirdata/P350/FTIR/FTIR.zip',
                           origin='omnic_export',
                           only=10,
                           delimiter = ';')

    A.plot_stack()
    show()

    print(A)
    assert A.shape == (10, 2843)


def test_read_csv_tg():
    B = NDDataset.read_csv('agirdata/P350/TGA/tg.csv',
                           directory=prefs.datadir,
                           origin='tga')
    assert B.shape == (3247,)
    print(B)
    B.plot()
    show()


def test_read_csv_IR():
    B = NDDataset.read_csv('irdata/IR.CSV', directory=prefs.datadir,
                           origin='ir', csv_delimiter=',')
    assert B.shape == (3736,)
    print(B)
    B.plot()
    show()

def test_read_without_directory():
    B = NDDataset.read_csv('irdata/IR.CSV')
    print(B)

def test_read_without_filename():
    B = NDDataset.read_csv()
    print(B)