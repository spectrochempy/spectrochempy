# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

import pytest
import numpy as np

import spectrochempy as scp
from pathlib import Path

RAMANDIR = scp.preferences.datadir / "ramandata"


@pytest.mark.skipif(
    not RAMANDIR.exists(),
    reason="Experimental data not available for testing",
)
def test_read_labspec():

    # single file
    nd = scp.read_labspec("Activation.txt", directory=RAMANDIR)
    assert nd.shape == (532, 1024)
    assert nd.y.dtype.name.startswith("datetime64")
    assert nd.comment == "Spectrum acquisition : 2016-05-01T11:27"

    # date has a different format
    nd = scp.read_labspec("serie190214-1.txt", directory=RAMANDIR)
    assert nd.shape == (168, 1024)
    assert nd.y.dtype.name.startswith("datetime64")
    assert nd.comment == "Spectrum acquisition : 2019-02-14T10:41"

    # read one dimensional
    nd = scp.read_labspec(RAMANDIR / "subdir" / "LiNbWO6-0-H.txt")
    assert nd.shape == (1, 1024)

    # with read_dir
    nd = scp.read_dir(directory=RAMANDIR / "subdir")
    assert nd.shape == (6, 1024)

    # empty txt file
    Path("i_am_empty.txt").touch()
    f = Path("i_am_empty.txt")
    nd = scp.read_labspec(f)
    f.unlink()
    assert nd is None

    # non labspec txt file
    f = Path("i_am_not_labspec.txt")
    f.write_text("blah")
    nd = scp.read_labspec(f)
    f.unlink()
    assert nd is None
