# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset


def test_read_write_json(IR_dataset_2D):
    ds = IR_dataset_2D

    f = ds.write('try2D.json')
    ds.read('try2D.json')
    dsr = scp.read('try2D.json')
    assert ds == dsr
    f.unlink()

    f = scp.write(ds, 'try2D.json')
    dsr = scp.read_json('try2D.json')
    assert ds == dsr

    dsr2 = NDDataset.read('try2D.json')  # NDDataset class method
    assert ds == dsr2

    f.unlink()


def test_write_nmr_to_json(NMR_dataset_1D):
    nd = NMR_dataset_1D
    nd.name = "nmr_1d"

    f = nd.write_json()
    assert f.name == 'nmr_1d.json'

    nd2 = scp.read_json('nmr_1d')
    assert nd2 == nd
