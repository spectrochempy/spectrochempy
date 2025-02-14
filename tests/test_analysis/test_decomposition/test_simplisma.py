# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

import pytest

import spectrochempy as scp
from spectrochempy.analysis.decomposition.simplisma import SIMPLISMA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils.plots import show


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_SIMPLISMA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.simplisma"
    chd.check_docstrings(
        module,
        obj=scp.SIMPLISMA,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


def test_simplisma():
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

    sma = SIMPLISMA(n_components=20, tol=0.2, noise=3, log_level="INFO")
    sma.fit(ds)

    sma.C.T.plot(title="Concentration")
    sma.St.plot(title="Components")
    sma.plotmerit(offset=0, nb_traces=10)

    # assert "3     29      29.0     0.0072     0.9981" in pure.logs

    show()
