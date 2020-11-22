# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os
import pytest
from pathlib import Path


import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy import general_preferences as prefs

# ......................................................................................................................
def test_read_opus():

    # single file
    A = NDDataset.read_opus(os.path.join('irdata', 'OPUS', 'test.0000'))
    assert A.shape == (1, 2567)
    assert A[0, 2303.8694].data == pytest.approx(2.72740, 0.00001)

    # using a window path
    A1 = NDDataset.read_opus('irdata\\OPUS\\test.0000')
    assert A1.shape == (1, 2567)

    # single file specified with pathlib
    datadir = Path(prefs.datadir)
    p = datadir / 'irdata' / 'OPUS' / 'test.0000'
    A2 = NDDataset.read_opus(p)
    assert A2.shape == (1, 2567)

    # multiple files not merged
    B = NDDataset.read_opus('test.0000', 'test.0001', 'test.0002', directory=os.path.join('irdata', 'OPUS'))
    assert isinstance(B, NDDataset)
    assert len(B) == 3

    # multiple files merged as the merge keyword is set to true
    C = scp.read_opus('test.0000', 'test.0001', 'test.0002', directory=os.path.join('irdata', 'OPUS'), merge=True)
    assert C.shape == (3, 2567)

    # multiple files to merge : they are passed as a list)
    D = NDDataset.read_opus(['test.0000', 'test.0001', 'test.0002'], directory=os.path.join('irdata', 'OPUS'))
    assert D.shape == (3, 2567)

    # multiple files not merged : they are passed as a list but merge is set to false
    E = scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory=os.path.join('irdata', 'OPUS'), merge=False)
    assert isinstance(E, list)
    assert len(E) == 3

    # read contents
    p = datadir / 'irdata' / 'OPUS' / 'test.0000'
    content = p.read_bytes()
    F = NDDataset.read_opus({p.name:content})
    assert F.name == p.name
    assert F.shape == (1, 2567)

    # read multiple contents
    l = [ datadir / 'irdata' / 'OPUS' / f'test.000{i}' for i in range(3)]
    G = NDDataset.read_opus({p.name : p.read_bytes() for p in l})
    assert len(G)==3

    # read multiple contents and merge them
    l = [ datadir / 'irdata' / 'OPUS' / f'test.000{i}' for i in range(3)]
    H = NDDataset.read_opus({p.name : p.read_bytes() for p in l}, merge=True)
    assert H.shape == (3, 2567)

    # read without filename -> open a dialog
    I = NDDataset.read_opus()

    # read in a directory (assume homogeneous type of data - else we must use the read function instead)
    I = NDDataset.read_opus(directory=datadir / 'irdata' / 'OPUS')
    assert len(I) == 4

    # again we can use merge to stack all 4 spectra
    J = NDDataset.read_opus(directory=datadir / 'irdata' / 'OPUS', merge=True)
    assert J.shape == (4, 2567)

    # single opus file using generic read function
    # if the protocol is given it is similar to the read_opus function
    F = NDDataset.read(os.path.join('irdata', 'OPUS', 'test.0000'), protocol='.opus')
    assert F.shape == (1, 2567)

    # No protocol?
    G = NDDataset.read(os.path.join('irdata', 'OPUS', 'test.0000'))
    assert G.shape == (1, 2567)
