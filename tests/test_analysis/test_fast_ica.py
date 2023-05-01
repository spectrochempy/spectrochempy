# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import os

import pytest

import spectrochempy as scp
from spectrochempy.analysis.fast_ica import FastICA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils.plots import show


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_FastICA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.fast_ica"
    chd.check_docstrings(
        module,
        obj=scp.FastICA,
        # exclude some errors - remove whatever you want to check
        exclude=["EX01", "SA01", "ES01", "PR06"],
    )


def test_fastICA():
    print("")
    data = NDDataset.read_matlab(
        os.path.join("matlabdata", "als2004dataset.MAT"), merge=False
    )
    print("Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):")
    print("")
    for mat in data:
        print("    " + mat.name, str(mat.shape))

    ds = data[-1]
    assert ds.name == "m1"

    ds.title = "absorbance"
    ds.units = "absorbance"
    ds.set_coordset(None, None)
    ds.y.title = "elution time"
    ds.x.title = "wavelength"
    ds.y.units = "hours"
    ds.x.units = "cm^-1"

    ica = FastICA(n_components=4, log_level="INFO")
    ica.fit(ds)

    ica.A.T.plot(title="Mixing matrix")
    ica.components.plot(title="Sources")
    ica.mixing.plot(title="mixing")
    ica.whitening.plot(title="whitening")
    ica.plotmerit(offset=0, nb_traces=10)

    # todo: complete testing (options, check methods...)
    show()
