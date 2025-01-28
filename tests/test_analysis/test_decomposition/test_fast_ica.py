# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

import pytest
from numpy.testing import assert_almost_equal
from sklearn.decomposition import FastICA as skl_ICA

import spectrochempy as scp
from spectrochempy.analysis.decomposition.fast_ica import FastICA as scp_ICA
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
@pytest.mark.skipif(
    os.name == "nt",
    reason="UnicodeDecodeError on github action with windows, but not locally",
)
def test_FastICA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.fast_ica"
    chd.check_docstrings(
        module,
        obj=scp.FastICA,
        # exclude some errors - remove whatever you want to check
        exclude=["EX01", "SA01", "ES01", "PR06"],
    )


def test_fastICA():
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

    ds_ = ds.data.copy()

    ica = scp_ICA(
        n_components=4, random_state=123, whiten="unit-variance", log_level="INFO"
    )
    ica.fit(ds)

    ica_ = skl_ICA(n_components=4, random_state=123, whiten="unit-variance")
    ica_.fit(ds_)

    # compare scpy and sklearn FastICA attributes
    assert_almost_equal(ica.components.data, ica_.components_)
    assert_almost_equal(ica.mixing.data, ica_.mixing_)
    assert_almost_equal(ica.mean.data, ica_.mean_)
    assert_almost_equal(ica.whitening.data, ica_.whitening_)
    assert ica.n_iter == ica_.n_iter_

    # compare scpy and sklearn FastICA methods
    U = ica.transform()
    U_ = ica_.transform(ds_)
    assert_almost_equal(U.data, U_)

    U = ica.fit_transform(ds)
    U_ = ica_.fit_transform(ds_)
    assert_almost_equal(U.data, U_)

    dshat = ica.inverse_transform()
    dshat_ = ica_.inverse_transform(U_)
    assert_almost_equal(dshat.data, dshat_)

    # test plots
    ica.A.T.plot(title="Mixing system A / ica.transform() ")
    ica.St.plot(title="Source spectra profiles / ica.mixing.T")
    ica.components.plot(title="Components / W / unmixing matrix")
    ica.mixing.plot(title="ica.mixing")
    ica.whitening.plot(title="ica.whitening")
    ica.plotmerit(offset=0, nb_traces=10)

    # Test masked data, x axis
    ica2 = scp.FastICA(n_components=4)
    ds[:, 10:20] = MASKED  # corn spectra, calibration
    ica2.fit(ds)
    #
    assert ica2._X.shape == (51, 86), "missing row or col should be removed"
    assert ica2.X.shape == (51, 96), "missing row or col restored"
    (
        assert_dataset_equal(ica2.X, ds, data_only=True),
        "input dataset should be reflected in the internal variable X (where mask is restored)",
    )

    ica2.plotmerit()

    # todo: complete testing of options
    show()
