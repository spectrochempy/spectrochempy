# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import pytest

import spectrochempy as scp
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import testing
from spectrochempy.utils.file import pathclean

irdatadir = pathclean(prefs.datadir) / "irdata"
cwd = Path.cwd()

try:
    from spectrochempy.core.common import dialogs
except ImportError:
    pytest.skip("dialogs not available with act", allow_module_level=True)


def test_write():
    nd = scp.read_omnic("irdata/nh4y-activation.spg")

    # API write methods needs an instance of a NDDataset as the first argument
    with pytest.raises(TypeError):
        scp.write()

    # the simplest way to save a dataset, is to use the function write with a filename as argument
    if (cwd / "essai.scp").exists():
        (cwd / "essai.scp").unlink()

    filename = nd.write("essai.scp")  # should not open a DIALOG
    assert filename == cwd / "essai.scp"
    assert filename.exists()

    # try to write it again
    filename = nd.write("essai.scp")  # should open a DIALOG to confirm

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
