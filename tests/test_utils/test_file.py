# -*- coding: utf-8 -*-

#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from pathlib import Path
from os import environ
from os.path import join

import pytest

from spectrochempy.core import info_, general_preferences as prefs
from spectrochempy.api import NO_DISPLAY
from spectrochempy.utils import get_filename, check_filenames
from spectrochempy.core.dataset.nddataset import NDDataset


def test_get_filename():
    # should read in the default prefs.datadir (and for testing we fix the name to environ['TEST_FILE']
    f = get_filename(filetypes=["OMNIC files (*.sp*)",
                                "SpectroChemPy files (*.scp)",
                                "all files (*)"])
    assert isinstance(f, dict)

    f = get_filename(filetypes=["OMNIC files (*.sp*)",
                                "SpectroChemPy files (*.scp)",
                                "all files (*)"],
                     dictionary=False)
    assert isinstance(f, list)
    assert isinstance(f[0], Path)
    if NO_DISPLAY:
        assert str(f[0]) == join(prefs.datadir, environ['TEST_FILE'])

    # directory specified by a keyword as well as the filename
    f = get_filename("nh4y.scp", directory="irdata")
    assert f == {'.scp': [Path('/Users/christian/Dropbox/SCP/spectrochempy/scp_data/testdata/irdata/nh4y.scp')]}

    # directory specified in the filename as a subpath of the data directory
    f = get_filename("irdata/nh4y.scp")
    assert f == {'.scp': [Path('/Users/christian/Dropbox/SCP/spectrochempy/scp_data/testdata/irdata/nh4y.scp')]}

    # no directory specified (filename must be in the working or the default  data directory
    f = get_filename("wodger.spg")

    # if it is not found an error is generated
    with pytest.raises(IOError):
        f = get_filename("nh4y.scp")

    # directory is implicit (we get every files inside, with an allowed extension)
    # WARNING:  Must end with a backslash
    f = get_filename("irdata/",
                     filetypes=['OMNIC files (*.spa, *.spg)','OMNIC series (*.srs)', 'all files (*.*)'],
                     listdir=True)

    assert len(f.keys()) == 3

    # should raise an error
    with pytest.raises(IOError):
        get_filename("~/xxxx",
                     filetypes=["OMNIC files (*.sp*)",
                                "SpectroChemPy files (*.scp)",
                                "all files (*)"])

def test_check_filenames():

    f1 = check_filenames("irdata",
                     filetypes=['OMNIC files (*.spa, *.spg)','OMNIC series (*.srs)', 'all files (*.*)'],
                     listdir=True)
    f2 = get_filename("irdata",
                     filetypes=['OMNIC files (*.spa, *.spg)','OMNIC series (*.srs)', 'all files (*.*)'],
                     )

    #assert len(f1.keys()) == 3
    #assert f1 == f
