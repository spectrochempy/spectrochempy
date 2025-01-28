# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from os import environ
from pathlib import Path

import pytest

from spectrochempy import NO_DISPLAY
from spectrochempy.core import preferences as prefs
from spectrochempy.utils.file import check_filenames, get_filenames, pathclean


def test_pathclean():
    # Using unix/mac way to write paths
    filename = pathclean("irdata/nh4y-activation.spg")
    assert filename.suffix == ".spg"
    assert filename.parent.name == "irdata"

    # or Windows
    filename = pathclean("irdata\\\\nh4y-activation.spg")
    assert filename.parent.name == "irdata"

    # Due to the escape character \\ in Unix, path string should be escaped \\\\
    # or the raw-string prefix `r` must be used as shown below
    filename = pathclean(r"irdata\\nh4y-activation.spg")
    assert filename.suffix == ".spg"
    assert filename.parent.name == "irdata"

    # of course should work if input is alreadya Path
    filename = pathclean(prefs.datadir / "irdata/nh4y-activation.spg")
    assert filename.suffix == ".spg"


def test_get_filename():
    # should read in the default prefs.datadir (and for testing we fix the name to environ['TEST_FILE']
    f = get_filenames(
        filetypes=["OMNIC files (*.spg *.spa *.srs)", "SpectroChemPy files (*.scp)"]
    )
    assert isinstance(f, dict)

    f = get_filenames(
        filetypes=["OMNIC files (*.spg *.spa *.srs)", "SpectroChemPy files (*.scp)"],
        dictionary=False,
    )
    assert isinstance(f, list)
    assert isinstance(f[0], Path)

    if NO_DISPLAY:
        assert f[0] == prefs.datadir / environ["TEST_FILE"]

    # directory specified by a keyword as well as the filename
    f = get_filenames("nh4y-activation.spg", directory="irdata")
    assert f == {".spg": [Path(prefs.datadir) / "irdata" / "nh4y-activation.spg"]}

    # directory specified in the filename as a subpath of the data directory
    f = get_filenames("irdata/nh4y-activation.spg")
    assert f == {".spg": [Path(prefs.datadir) / "irdata" / "nh4y-activation.spg"]}

    # no directory specified (filename must be in the working or the default  data directory
    f = get_filenames("wodger.spg")

    # if it is not found an error is generated
    with pytest.raises(IOError):
        f = get_filenames("nh4y-activation.spg")

    # directory is implicit (we get every files inside, with an allowed extension)
    # WARNING:  Must end with a backslash
    f = get_filenames(
        "irdata/",
        filetypes=[
            "OMNIC files (*.spa, *.spg)",
            "OMNIC series (*.srs)",
            "all files (*.*)",
        ],
        listdir=True,
    )
    if ".scp" in f.keys():
        del f[".scp"]
    assert len(f.keys()) == 2

    # should raise an error
    with pytest.raises(IOError):
        get_filenames(
            "~/xxxx",
            filetypes=[
                "OMNIC files (*.sp*)",
                "SpectroChemPy files (*.scp)",
                "all files (*)",
            ],
        )


def test_check_filename():
    filename = "irdata/nh4y-activation.spg"

    # return a dictionary (after opening a dialog)
    filenames = check_filenames()
    assert isinstance(filenames, dict)
    assert filenames == {".spg": [prefs.datadir / pathclean(filename)]}

    filenames = check_filenames(filename)
    assert isinstance(filenames, list)
    assert filenames[0] == prefs.datadir / pathclean(filename)

    filenames = check_filenames([filename])
    assert isinstance(filenames, list)
    assert filenames[0] == prefs.datadir / pathclean(filename)

    # return the dictionary itself
    filenames = check_filenames({"xxx": [filename]})
    assert isinstance(filenames, dict)
    assert filenames == {"xxx": [filename]}
