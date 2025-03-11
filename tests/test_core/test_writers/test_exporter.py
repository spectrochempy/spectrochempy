# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import testing
from spectrochempy.utils.file import pathclean

irdatadir = pathclean(prefs.datadir) / "irdata"


def test_write(mock_cwd):
    nd = scp.read_omnic("irdata/nh4y-activation.spg")

    # API write methods needs an instance of a NDDataset as the first argument
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'dataset'"
    ):
        scp.write()

    # the simplest way to save a dataset, is to use the function write with a filename as argument
    filename = nd.write("essai.scp")  # should not open a DIALOG
    assert filename == mock_cwd / "essai.scp"
    assert filename.exists()

    # try to write it again
    with pytest.raises(FileExistsError):
        nd.write("essai.scp")

    # write it again with overwrite
    filename = nd.write("essai.scp", overwrite=True)

    # Read the file and compare
    nd2 = NDDataset.load(filename)
    testing.assert_dataset_equal(nd2, nd)

    # we can also use the read method to read it
    nd3 = scp.read(filename)
    testing.assert_dataset_equal(nd3, nd)

    filename.unlink()

    # if the filename is omitted, write a file with the dataset name and the extension '.scp'
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
