# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os
import pathlib

import pytest

import spectrochempy.utils.exceptions
from spectrochempy import NDDataset  # , preferences as prefs
from spectrochempy.core import preferences as prefs
from spectrochempy.core.readers.importer import (
    ALIAS,
    FILETYPES,
    Importer,
    _importer_method,
    read,
    read_dir,
)
from spectrochempy.utils.file import pathclean

try:
    from spectrochempy.core.common import dialogs
except ImportError:
    pytest.skip("dialogs not available with act", allow_module_level=True)

DATADIR = prefs.datadir


# Simulation of a read function
def read_fake(*paths, **kwargs):
    kwargs["filetypes"] = ["FAKE files (*fk, *.fk1, .fk2)"]
    kwargs["protocol"] = ["fake", ".fk", "fk1", "fk2"]
    importer = Importer()

    return importer(*paths, **kwargs)


read_fk = read_fake
setattr(NDDataset, "read_fk", read_fk)


@_importer_method
def _read_fake(*args, **kwargs):
    dataset, filename = args
    content = kwargs.get("content", False)

    if content:
        dataset = fake_dataset(content=True)
    else:
        # if filename.exists():    This does not work with fs
        if os.path.exists(filename):
            if filename.stem == "otherfake":
                dataset = fake_dataset(size=6)
            elif filename.stem == "emptyfake":
                dataset = None
            else:
                dataset = fake_dataset()
        else:
            raise (FileNotFoundError)

    return dataset  # empty if file


@_importer_method
def _read_fk(*args, **kwargs):
    return Importer._read_fake(*args, **kwargs)


# Test of the Importer class


def fake_dataset(*args, size=3, **kwargs):
    if not args:
        ds = NDDataset([range(size)])
    else:
        ds = NDDataset(
            [[range(4)]],
        )
    return ds


def dialog_cancel(*args, **kwargs):
    # mock a dialog cancel action
    return None


def dialog_open(*args, **kwargs):
    # mock opening a dialog

    directory = kwargs.get("directory", None)
    if directory is None:
        directory = pathclean(DATADIR / "fakedir")

    if kwargs.get("filters") == "directory":
        return directory

    if not args and not kwargs.get("single"):
        return [DATADIR / "fakedir" / f"fake{i + 1}.fk" for i in range(2)]

    return [DATADIR / "fakedir" / f"fake{i + 1}.fk" for i in range(4)]


def directory_glob(*args, **kwargs):
    res = [DATADIR / f"fakedir/fake{i + 1}.fk" for i in range(4)]
    res.append(DATADIR / "fakedir/emptyfake.fk")
    if len(args) > 1 and args[1].startswith("**/"):
        # recursive
        res.append(DATADIR / "fakedir/subdir/fakesub1.fk")
    return res


def test_importer(monkeypatch, fs):
    fs.create_file("/var/data/xx1.txt")
    assert os.path.exists("/var/data/xx1.txt")

    # mock filesystem
    fs.create_dir(DATADIR)

    # try to read unexistent scp file
    f = DATADIR / "fakedir/fakescp.scp"
    with pytest.raises(FileNotFoundError):
        read(f, local_only=True)  # local_only to avoid remote search
        # (which do not work with the monkeypatch - problem with certificates)

    # make fake file
    fs.create_file(f)
    monkeypatch.setattr(NDDataset, "load", fake_dataset)

    nd = read(f, local_only=True)
    assert nd == fake_dataset(f)

    nd = read(f.stem, directory=DATADIR / "fakedir/", protocol="scp", local_only=True)
    assert nd == fake_dataset(f)

    nd = read(f.stem, directory=DATADIR / "fakedir/", local_only=True)
    assert nd == fake_dataset(f)

    # Generic read without parameters and dialog cancel
    monkeypatch.setattr(dialogs, "open_dialog", dialog_cancel)
    monkeypatch.setenv(
        "KEEP_DIALOGS", "True"
    )  # we ask to display dialogs as we will mock them.

    nd = read(local_only=True)
    assert nd is None

    # read as class method
    nd1 = NDDataset.read(local_only=True)
    assert nd1 is None

    # NDDataset instance as first arguments
    nd = NDDataset()
    nd2 = nd.read(local_only=True)
    assert nd2 is None

    nd = read(default_filter="matlab", local_only=True)
    assert nd is None

    # Check if Filetype is not known
    f = DATADIR / "fakedir/not_exist_fake.fk"
    with pytest.raises(TypeError):
        read_fake(f, local_only=True)

    # Make fake type acceptable
    FILETYPES.append(("fake", "FAKE files (*.fk)"))
    ALIAS.append(("fk", "fake"))
    monkeypatch.setattr("spectrochempy.core.readers.importer.FILETYPES", FILETYPES)
    monkeypatch.setattr("spectrochempy.core.readers.importer.ALIAS", ALIAS)

    # Check not existing filename
    f = DATADIR / "fakedir/not_exist_fake.fk"
    with pytest.raises(FileNotFoundError):
        read_fake(f, local_only=True)

    # Generic read with a wrong protocol
    with pytest.raises(spectrochempy.utils.exceptions.ProtocolError):
        read(f, protocol="wrongfake", local_only=True)

    # Generic read with a wrong file extension
    with pytest.raises(TypeError):
        g = DATADIR / "fakedir/otherfake.farfelu"
        read(g, local_only=True)

    # Mock file
    f = DATADIR / "fakedir/fake.fk"
    fs.create_file(f)

    # specific read_(protocol) function
    nd = read_fk(f, local_only=True)
    assert nd == fake_dataset()

    # should also be a Class function
    nd = NDDataset.read_fk(f, local_only=True)
    assert nd == fake_dataset()

    # and a NDDataset instance function
    nd = NDDataset().read_fk(f, local_only=True)
    assert nd == fake_dataset()

    # single file without protocol inferred from filename
    nd = read(f, local_only=True)
    assert nd == fake_dataset()

    # single file read with protocol specified
    nd = read(f, protocol="fake", local_only=True)
    assert nd == fake_dataset()

    # attribute a new name
    nd = read(f, name="toto", local_only=True)
    assert nd.name == "toto"

    # mock some fake file and assume they exists
    f1 = DATADIR / "fakedir/fake1.fk"
    f2 = DATADIR / "fakedir/fake2.fk"
    f3 = DATADIR / "fakedir/fake3.fk"
    f4 = DATADIR / "fakedir/fake4.fk"
    f5 = DATADIR / "fakedir/otherdir/otherfake.fk"
    f6 = DATADIR / "fakedir/emptyfake.fk"  # return None when reader
    fs.create_file(f1)
    fs.create_file(f2)
    fs.create_file(f3)
    fs.create_file(f4)
    fs.create_file(f5)
    fs.create_file(f6)
    # l = list(pathclean("/Users/christian/test_data/fakedir").iterdir())

    # multiple compatible 1D files automatically merged
    nd = read(f1, f2, f3, local_only=True)
    assert nd.shape == (3, 3)

    nd = read([f1, f2, f3], name="fake_merged", local_only=True)
    assert nd.shape == (3, 3)
    assert nd.name == "fake_merged"

    # multiple compatible 1D files not merged if the merge keyword is set to False
    nd = read([f1, f2, f3], names=["a", "c", "b"], merge=False, local_only=True)
    assert isinstance(nd, list)
    assert len(nd) == 3 and nd[0] == fake_dataset()
    assert nd[1].name == "c"

    # do not merge inhomogeneous dataset
    nd = read([f1, f2, f5], local_only=True)
    assert isinstance(nd, list)

    # too short list of names.  Not applied
    nd = read([f1, f2, f3], names=["a", "c"], merge=False, local_only=True)
    assert nd[0].name.startswith("NDDataset")

    monkeypatch.setattr(spectrochempy.core.common.dialogs, "open_dialog", dialog_open)
    nd = read(
        local_only=True
    )  # should open a dialog (but to selects individual filename (here only simulated)
    assert nd.shape == (2, 3)

    # read in a directory
    monkeypatch.setattr(pathlib.Path, "glob", directory_glob)

    # directory selection
    nd = read(protocol="fake", directory=DATADIR / "fakedir", local_only=True)
    assert nd.shape == (4, 3)

    nd = read(
        protocol="fake", directory=DATADIR / "fakedir", merge=False, local_only=True
    )
    assert len(nd) == 4
    assert isinstance(nd, list)

    nd = read(iterdir=True, directory=DATADIR / "fakedir", local_only=True)
    assert len(nd) == 4
    assert not isinstance(nd, list)

    # if a directory is passed as a keyword, the behavior is different:
    # a dialog for file selection occurs except if iterdir is set to True
    nd = read(directory=DATADIR / "fakedir", iterdir=False, local_only=True)
    assert nd.shape == (2, 3)  # -> file selection dialog

    nd = read(directory=DATADIR / "fakedir", iterdir=True, local_only=True)
    assert nd.shape == (4, 3)  # -> directory selection dialog

    # read_dir()

    nd = read_dir(DATADIR / "fakedir", local_only=True)
    assert nd.shape == (4, 3)

    nd1 = read_dir(local_only=True)
    assert nd1 == nd

    nd = read_dir(
        directory=DATADIR / "fakedir", local_only=True
    )  # open a dialog to eventually select directory inside the specified
    # one
    assert nd.shape == (4, 3)

    fs.create_file(DATADIR / "fakedir/subdir/fakesub1.fk")
    nd = read_dir(directory=DATADIR / "fakedir", recursive=True, local_only=True)
    assert nd.shape == (5, 3)

    # no merging
    nd = read_dir(
        directory=DATADIR / "fakedir", recursive=True, merge=False, local_only=True
    )
    assert len(nd) == 5
    assert isinstance(nd, list)

    # Simulate reading a content
    nd = read({"somename.fk": "a fake content"})
    assert nd == fake_dataset(content=True)
    nd = read_fake({"somename.fk": "a fake content"})
    assert nd == fake_dataset(content=True)

    # read multiple contents and merge them
    nd = read(
        {
            "somename.fk": "a fake content",
            "anothername.fk": "another fake content",
            "stillanothername.fk": "still another fake content",
        }
    )
    assert nd.shape == (3, 3)

    # do not merge
    nd = read(
        {"somename.fk": "a fake content", "anothername.fk": "another fake content"},
        merge=False,
    )
    assert isinstance(nd, list)
    assert len(nd) == 2
