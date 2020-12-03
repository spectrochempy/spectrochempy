# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset


def test_read_write_json(IR_dataset_2D):
    ds = IR_dataset_2D

    ds.write('try2D.json')
    ds.read('try2D.json')

    dsr = scp.read('try2D.json')
    assert ds == dsr
    os.remove('try2D.json')

    scp.write(ds, 'try2D.json')
    dsr = scp.read('try2D.json')
    assert ds == dsr

    dsr2 = NDDataset.read('try2D.json')  # NDDataset class method
    assert ds == dsr2

    os.remove('try2D.json')

    # #write to string
    # s = ds.write(to_string=True, protocol='json')
    # assert s.startswith('{"data": {"serialized": "gASVhAAA')
    #
    # s = ds.write(to_string=True) # json by default
    # assert s.startswith('{"data": {"serialized": "gASVhAAA')
