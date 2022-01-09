# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path

import pytest

import spectrochempy as scp
from spectrochempy import NDDataset, preferences as prefs
from spectrochempy.utils import pathclean, testing

irdatadir = pathclean(prefs.datadir) / "irdata"
cwd = Path.cwd()


# ..............................................................................
def test_write():
    nd = scp.read_omnic("irdata/nh4y-activation.spg")

    # API write methods needs an instance of a NDDataset as the first argument
    with pytest.raises(TypeError):
        scp.write()

    # the simplest way to save a dataset, is to use the function write with a filename as argument
    filename = nd.write("essai.scp")
    assert filename == cwd / "essai.scp"

    nd2 = NDDataset.load(filename)
    testing.assert_dataset_equal(nd2, nd)
    filename.unlink()

    # if the filename is omitted, a dialog is opened to select a name (and a protocol)
    filename = nd.write()
    assert filename is not None
    assert filename.stem == nd.name
    assert filename.suffix == ".scp"
    filename.unlink()

    # # a write protocole can be specified
    # filename = nd.write(protocole='json')
    # assert filename is not None
    # assert filename.stem == nd.name
    # assert filename.suffix == '.json'
    # filename.unlink()

    irdatadir = pathclean(prefs.datadir) / "irdata"
    for f in ["essai.scp", "nh4y-activation.scp"]:
        if (irdatadir / f).is_file():
            (irdatadir / f).unlink()


# EOF
