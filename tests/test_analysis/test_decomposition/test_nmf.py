# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the NMF module

"""

import os

import pytest
from numpy.testing import assert_almost_equal
from sklearn.decomposition import NMF as skl_NMF

import spectrochempy as scp
from spectrochempy.analysis.decomposition.nmf import NMF
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.plots import show
from spectrochempy.utils.testing import assert_dataset_equal


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_NMF_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.nmf"
    chd.check_docstrings(
        module,
        obj=scp.NMF,
        # exclude some errors - remove whatever you want to check
        exclude=["EX01", "SA01", "ES01", "PR06"],
    )


def test_nmf():
    # Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110))
    ds = NDDataset.read_matlab(
        os.path.join("matlabdata", "als2004dataset.MAT"), merge=False
    )[-1]

    ds.title = "absorbance"
    ds.units = "absorbance"
    ds.set_coordset(None, None)
    ds.y.title = "elution time"
    ds.x.title = "wavelength"
    ds.y.units = "hours"
    ds.x.units = "au"

    ds = ds.clip(a_min=0)
    ds_ = ds.data.copy()

    nmf = NMF(n_components=4, random_state=123, log_level="INFO")
    nmf.fit(ds)

    nmf_ = skl_NMF(n_components=4, random_state=123)
    nmf_.fit(ds_)

    # compare scpy and sklearn NMF attributes
    assert_almost_equal(nmf.components.data, nmf_.components_)

    # compare scpy and sklearn NMF methods
    U = nmf.transform()
    U_ = nmf_.transform(ds_)
    assert_almost_equal(U.data, U_)

    U = nmf.fit_transform(ds)
    U_ = nmf_.fit_transform(ds_)
    assert_almost_equal(U.data, U_, decimal=3)

    dshat = nmf.inverse_transform()
    dshat_ = nmf_.inverse_transform(U_)
    assert_almost_equal(dshat.data, dshat_, decimal=4)

    # test plots
    U.T.plot(title="nmf.transform() ")
    nmf.components.plot(title="components")
    nmf.plotmerit(offset=0, nb_traces=10)

    # Test masked data, x axis
    nmf2 = NMF(n_components=4)
    ds[:, 10:20] = MASKED  # corn spectra, calibration
    nmf2.fit(ds)
    #
    assert nmf2._X.shape == (51, 86), "missing row or col should be removed"
    assert nmf2.X.shape == (51, 96), "missing row or col restored"
    (
        assert_dataset_equal(nmf2.X, ds, data_only=True),
        "input dataset should be reflected in the internal variable X (where mask is restored)",
    )

    nmf2.plotmerit()

    # todo: complete testing of options
    show()
