# -*- coding: utf-8 -*-
# flake8: noqa


import os
from pathlib import Path

import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy import preferences as prefs


# ..............................................................................
def test_read():
    f = Path("irdata/OPUS/test.0000")

    A1 = NDDataset.read_opus(f)
    assert A1.shape == (1, 2567)

    # single file read with protocol specified
    A2 = NDDataset.read(f, protocol="opus")
    assert A2 == A1

    A3 = scp.read("irdata/nh4y-activation.spg", protocol="omnic")
    assert str(A3) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # single file without protocol
    # inferred from filename
    A4 = NDDataset.read(f)
    assert A4 == A1

    A5 = scp.read("irdata/nh4y-activation.spg")
    assert str(A5) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # native format
    f = A5.save_as("nh4y.scp")
    A6 = scp.read("irdata/nh4y.scp")
    assert str(A6) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    A7 = scp.read("nh4y", directory="irdata", protocol="scp")
    assert str(A7) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    A8 = scp.read("nh4y", directory="irdata")
    assert str(A8) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    f.unlink()

    # multiple compatible 1D files automatically merged
    B = NDDataset.read(
        "test.0000", "test.0001", "test.0002", directory=os.path.join("irdata", "OPUS")
    )
    assert str(B) == "NDDataset: [float64] a.u. (shape: (y:3, x:2567))"
    assert len(B) == 3

    # multiple compatible 1D files not merged if the merge keyword is set to False
    C = scp.read(
        "test.0000",
        "test.0001",
        "test.0002",
        directory=os.path.join("irdata", "OPUS"),
        merge=False,
    )
    assert isinstance(C, list)

    # multiple 1D files to merge
    D = NDDataset.read(
        ["test.0000", "test.0001", "test.0002"],
        directory=os.path.join("irdata", "OPUS"),
    )
    assert D.shape == (3, 2567)

    # multiple 1D files not merged : they are passed as a list but merge is set to false
    E = scp.read(
        ["test.0000", "test.0001", "test.0002"],
        directory=os.path.join("irdata", "OPUS"),
        merge=False,
    )
    assert isinstance(E, list)
    assert len(E) == 3

    # read contents
    datadir = Path(prefs.datadir)
    p = datadir / "irdata" / "OPUS" / "test.0000"
    content = p.read_bytes()
    F = NDDataset.read({p.name: content})
    assert F.name == p.name
    assert F.shape == (1, 2567)

    # read multiple 1D contents and merge them
    lst = [datadir / "irdata" / "OPUS" / f"test.000{i}" for i in range(3)]
    G = NDDataset.read({p.name: p.read_bytes() for p in lst})
    assert G.shape == (3, 2567)
    assert len(G) == 3

    # read multiple  1D contents awithout merging
    lst = [datadir / "irdata" / "OPUS" / f"test.000{i}" for i in range(3)]
    H = NDDataset.read({p.name: p.read_bytes() for p in lst}, merge=False)
    isinstance(H, list)
    assert len(H) == 3

    filename = datadir / "wodger.spg"
    content = filename.read_bytes()

    # change the filename to be sure that the file will be read from the passed content
    filename = "try.spg"

    # The most direct way to pass the byte content information
    nd = NDDataset.read(filename, content=content)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:2, x:5549))"

    # It can also be passed using a dictionary structure {filename:content, ....}
    nd = NDDataset.read({filename: content})
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:2, x:5549))"

    # Case where the filename is not provided
    nd = NDDataset.read(content)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:2, x:5549))"

    # Try with an .spa file
    filename = datadir / "irdata/subdir/7_CZ0-100 Pd_101.SPA"
    content = filename.read_bytes()
    filename = "try.spa"

    filename2 = datadir / "irdata/subdir/7_CZ0-100 Pd_102.SPA"
    content2 = filename2.read_bytes()
    filename = "try2.spa"

    nd = NDDataset.read({filename: content})
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:1, x:5549))"

    # Try with only a .spa content
    nd = NDDataset.read(content)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:1, x:5549))"

    # Try with several .spa content (should be stacked into a single nddataset)
    nd = NDDataset.read({filename: content, filename2: content2})
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:2, x:5549))"

    nd = NDDataset.read(content, content2)
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:2, x:5549))"


def test_generic_read():
    # filename + extension specified
    ds = scp.read("wodger.spg")
    assert ds.name == "wodger"

    # save with no filename (should save wodger.scp)
    path = ds.save()

    assert isinstance(path, Path)
    assert path.stem == ds.name
    assert path.parent == ds.directory
    assert path.suffix == ".scp"

    # read should be équivalent to load (but read is a more general function,
    dataset = NDDataset.load("wodger.scp")
    assert dataset.name == "wodger"


def test_read_dir():
    datadir = Path(prefs.datadir)

    A = scp.read()  # should open a dialog (but to selects individual filename

    # if we want the whole dir  - listdir must be used
    # this is equivalent to read_dir with a dialog to select directories only
    A = scp.read(listdir=True, directory=datadir / "irdata" / "subdir")
    assert len(A) == 4
    A1 = scp.read_dir(directory=datadir / "irdata" / "subdir")
    assert A == A1

    # listdir is not necessary if a directory location is given as a single argument
    B = scp.read(datadir / "irdata" / "subdir", listdir=True)
    B1 = scp.read(datadir / "irdata" / "subdir")
    assert B == B1

    # if a directory is passed as a keyword, the behavior is different:
    # a dialog for file selection occurs except if listdir is set to True
    scp.read(
        directory=datadir / "irdata" / "subdir", listdir=True
    )  # -> file selection dialog
    scp.read(
        directory=datadir / "irdata" / "subdir", listdir=True
    )  # -> directory selection dialog
