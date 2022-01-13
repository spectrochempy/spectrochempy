# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path
import os

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import preferences as prefs


def test_read_omnic():
    # Class method opening a dialog (but for test it is preset)
    nd = NDDataset.read_omnic()
    assert nd.name == "nh4y-activation"

    # class method
    nd1 = scp.NDDataset.read_omnic("irdata/nh4y-activation.spg")
    assert nd1.title == "absorbance"
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # API method
    nd2 = scp.read_omnic("irdata/nh4y-activation.spg")
    assert nd1 == nd2

    # opening list of dataset
    l1 = scp.read_omnic("wodger.spg", "irdata/nh4y-activation.spg")
    assert len(l1) == 2
    assert str(l1[0]) == "NDDataset: [float64] a.u. (shape: (y:2, x:5549))"
    assert str(l1[1]) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # It is also possible to use more specific reader function such as
    # `read_spg`, `read_spa` or `read_srs` - they are alias of the read_omnic function.
    l2 = scp.read_spg("wodger.spg", "irdata/nh4y-activation.spg")
    assert len(l2) == 2

    # pathlib.Path objects can be used instead of string for filenames
    p = Path("wodger.spg")
    nd3 = scp.read_omnic(p)
    assert nd3 == l1[0]

    # merging
    nd = scp.read_omnic(["wodger.spg", "irdata/nh4y-activation.spg"])
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:57, x:5549))"

    # merging
    nd = scp.read_omnic("wodger.spg", "irdata/nh4y-activation.spg", merge=True)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:57, x:5549))"


def test_read_omnic_dir():
    # Read in a directory (assume that only OMNIC files are present in the directory
    #        (else we must use the generic `read` function instead)

    nd = scp.read_omnic("irdata/subdir/1-20")
    assert isinstance(nd, NDDataset)
    assert len(nd) == 3

    # we can use merge=False
    lst = scp.read_omnic("irdata/subdir/1-20", merge=False)
    assert isinstance(lst, list)
    assert len(lst) == 3


def test_read_omnic_contents():
    # test read_omnic with byte spg content
    datadir = prefs.datadir
    filename = "wodger.spg"
    with open(os.path.join(datadir, filename), "rb") as fil:
        content = fil.read()
    nd1 = scp.read_omnic(filename)
    nd2 = scp.read_omnic({filename: content})
    assert nd1 == nd2

    # Test bytes contents for spa files
    datadir = prefs.datadir
    filename = "7_CZ0-100 Pd_101.SPA"
    with open(os.path.join(datadir, "irdata", "subdir", filename), "rb") as fil:
        content = fil.read()
    nd = NDDataset.read_omnic({filename: content})
    assert nd.shape == (1, 5549)

    # test read_omnic with several contents
    datadir = prefs.datadir
    filename1 = "7_CZ0-100 Pd_101.SPA"
    with open(os.path.join(datadir, "irdata", "subdir", filename1), "rb") as fil:
        content1 = fil.read()
    filename2 = "wodger.spg"
    with open(os.path.join(datadir, filename2), "rb") as fil:
        content2 = fil.read()
    listnd = NDDataset.read_omnic(
        {filename1: content1, filename2: content2}, merge=True
    )
    assert listnd.shape == (3, 5549)


def test_read_spa():
    nd = scp.read_spa("irdata/subdir/20-50/7_CZ0-100 Pd_21.SPA")
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:1, x:5549))"

    nd = scp.read_spa("irdata/subdir", merge=True)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:4, x:5549))"

    nd = scp.read_spa("irdata/subdir", merge=True, recursive=True)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:8, x:5549))"

    lst = scp.read("irdata", merge=True, recursive=True)  # not selective on extension
    assert isinstance(lst, list)
    assert len(lst) >= 88


def test_read_srs():
    a = scp.read("irdata/omnic series/rapid_scan.srs")
    assert str(a) == "NDDataset: [float64] V (shape: (y:643, x:4160))"

    b = scp.read("irdata/omnic series/rapid_scan_reprocessed.srs")
    assert str(b) == "NDDataset: [float64] a.u. (shape: (y:643, x:3734))"

    c = scp.read("irdata/omnic series/GC Demo.srs")
    assert c == []

    d = scp.read("irdata/omnic series/TGA demo.srs")
    assert d == []
