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
def test_read():

    f = Path('irdata/OPUS/test.0000')

    A1 = NDDataset.read_opus(f)
    assert A1.shape == (1, 2567)

    # single file read with protocol specified
    A2 = NDDataset.read(f, protocol='opus')
    assert A2 == A1

    A3 = scp.read('irdata/nh4y-activation.spg', protocol='omnic')
    assert str(A3) == 'NDDataset: [float32] a.u. (shape: (y:55, x:5549))'

    # single file without protocol
    # inferred from filename
    A4 = NDDataset.read(f)
    assert A4==A1

    A5 = scp.read('irdata/nh4y-activation.spg')
    assert str(A5) == 'NDDataset: [float32] a.u. (shape: (y:55, x:5549))'

    # native format
    A6 = scp.read('irdata/nh4.scp')
    assert str(A6) == 'NDDataset: [float32] a.u. (shape: (y:55, x:5549))'

    A7 = scp.read('nh4', directory='irdata', protocol='.scp')
    assert str(A7) == 'NDDataset: [float32] a.u. (shape: (y:55, x:5549))'

    A8 = scp.read('nh4', directory='irdata')
    assert str(A8) == 'NDDataset: [float32] a.u. (shape: (y:55, x:5549))'

    # multiple files not merged
    B = NDDataset.read('test.0000', 'test.0001', 'test.0002', directory=os.path.join('irdata', 'OPUS'))
    assert isinstance(B, list)
    assert len(B) == 3

    # multiple files merged as the merge keyword is set to true
    C = scp.read('test.0000', 'test.0001', 'test.0002', directory=os.path.join('irdata', 'OPUS'), merge=True)
    assert C.shape == (3, 2567)

    # multiple files to merge : they are passed as a list)
    D = NDDataset.read(['test.0000', 'test.0001', 'test.0002'], directory=os.path.join('irdata', 'OPUS'))
    assert D.shape == (3, 2567)

    # multiple files not merged : they are passed as a list but merge is set to false
    E = scp.read(['test.0000', 'test.0001', 'test.0002'], directory=os.path.join('irdata', 'OPUS'), merge=False)
    assert isinstance(E, list)
    assert len(E) == 3

    # read contents
    datadir = Path(prefs.datadir)
    p = datadir / 'irdata' / 'OPUS' / 'test.0000'
    content = p.read_bytes()
    F = NDDataset.read({p.name:content})
    assert F.name == p.name
    assert F.shape == (1, 2567)

    # read multiple contents
    l = [ datadir / 'irdata' / 'OPUS' / f'test.000{i}' for i in range(3)]
    G = NDDataset.read({p.name : p.read_bytes() for p in l})
    assert len(G)==3

    # read multiple contents and merge them
    l = [ datadir / 'irdata' / 'OPUS' / f'test.000{i}' for i in range(3)]
    H = NDDataset.read({p.name : p.read_bytes() for p in l}, merge=True)
    assert H.shape == (3, 2567)

    filename = datadir / 'wodger.spg'
    content = filename.read_bytes()

    # change the filename to be sure that the file will be read from the passed content
    filename = 'try.spg'

    # The most direct way to pass the byte content information
    nd = NDDataset.read(filename, content=content)
    assert str(nd) == 'NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    # It can also be passed using a dictionary structure {filename:content, ....}
    nd = NDDataset.read({filename:content})
    assert str(nd) == 'NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    # Case where the filename is not provided
    nd = NDDataset.read(content)
    assert str(nd) == 'NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    # Try with an .spa file
    filename = datadir / 'irdata/subdir/7_CZ0-100 Pd_101.SPA'
    content = filename.read_bytes()
    filename = 'try.spa'

    filename2 = datadir / 'irdata/subdir/7_CZ0-100 Pd_102.SPA'
    content2 = filename2.read_bytes()
    filename = 'try2.spa'

    nd = NDDataset.read({filename:content})
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:1, x:5549))'

    # Try with only a .spa content
    nd = NDDataset.read(content)
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:1, x:5549))'

    # Try with several .spa content (should be stacked into a single nddataset)
    nd = NDDataset.read({filename:content, filename2:content2}, merge=True)
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    nd = NDDataset.read(content, content2, merge=True)
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:2, x:5549))'


def test_generic_read():
    import time

    # filename + extension specified
    start = time.time()
    ds = NDDataset.read('wodger.spg')
    t1 = (time.time() - start)
    assert ds.name == 'wodger'

    # save with no filename (should save wodger.scp)
    path = ds.save()

    assert isinstance(path, Path)
    assert path.stem == ds.name
    assert path.parent == ds.directory

    # should be équivalent to load (but read is a more general function
    start = time.time()
    dataset = NDDataset.read('wodger.scp')
    t2 =(time.time() - start)

    p =  (t2 - t1) * 100./ t1
    assert p<0


