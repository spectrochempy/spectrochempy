# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the NMF module

"""

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.decomposition import NMF as skl_NMF

import spectrochempy as scp
from spectrochempy.analysis.decomposition.nmf import NMF
from spectrochempy.utils import docutils as chd
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.testing import assert_dataset_equal


NMF_NONNEGATIVE_TOL = 1.0e-12


@pytest.fixture()
def nmf_dataset():
    elution_time = np.linspace(0.0, 1.0, 18)
    wavelength = np.linspace(400.0, 760.0, 20)

    concentrations = np.column_stack(
        [
            1.2 * np.exp(-0.5 * ((elution_time - 0.25) / 0.10) ** 2),
            0.9 * np.exp(-0.5 * ((elution_time - 0.52) / 0.13) ** 2),
            1.1 * np.exp(-0.5 * ((elution_time - 0.78) / 0.11) ** 2),
        ]
    )
    spectra = np.vstack(
        [
            0.2 + np.exp(-0.5 * ((wavelength - 470.0) / 35.0) ** 2),
            0.1 + 0.8 * np.exp(-0.5 * ((wavelength - 585.0) / 45.0) ** 2),
            0.15 + 0.9 * np.exp(-0.5 * ((wavelength - 690.0) / 40.0) ** 2),
        ]
    )
    data = concentrations @ spectra

    return scp.NDDataset(
        data,
        coordset=[
            scp.Coord(elution_time, title="elution time", units="hours"),
            scp.Coord(wavelength, title="wavelength", units="nm"),
        ],
        title="synthetic NMF mixture",
        units="absorbance",
    )


@pytest.fixture()
def nmf_model():
    return NMF(
        n_components=3,
        init="nndsvda",
        max_iter=1000,
        random_state=123,
        tol=1.0e-8,
    )


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_NMF_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.nmf"

    # Base exclusions for all Python versions
    exclude = ["EX01", "SA01", "ES01", "PR01", "PR06"]

    # Temporary workaround for Python 3.11 numpydoc/docstring-generation
    # inconsistencies. PR01 errors (parameters not documented) appear on
    # Python 3.11 due to differences in generated docstrings.
    # Validation remains strict on Python 3.12+.
    # TODO: Revisit when Python 3.11 support is dropped or numpydoc is updated.
    if sys.version_info[:2] == (3, 11):
        exclude += ["PR01"]

    chd.check_docstrings(
        module,
        obj=scp.NMF,
        exclude=exclude,
    )


def test_nmf_fit_components_and_metadata(nmf_dataset, nmf_model):
    result = nmf_model.fit(nmf_dataset)

    assert result is nmf_model
    assert nmf_model._X.shape == nmf_dataset.shape
    assert_dataset_equal(nmf_model.X, nmf_dataset)
    assert nmf_model.components.shape == (3, nmf_dataset.shape[1])
    assert nmf_model.components.dims == ["k", "x"]
    assert nmf_model.components.title == nmf_dataset.title
    assert nmf_model.components.x.title == nmf_dataset.x.title
    assert nmf_model.components.x.units == nmf_dataset.x.units
    assert np.all(nmf_model.components.data >= -NMF_NONNEGATIVE_TOL)


def test_nmf_matches_sklearn_on_synthetic_data(nmf_dataset, nmf_model):
    dataset = nmf_dataset
    nmf_model.fit(dataset)

    expected = skl_NMF(
        n_components=3,
        init="nndsvda",
        max_iter=1000,
        random_state=123,
        tol=1.0e-8,
    )
    expected.fit(dataset.data)

    assert_allclose(nmf_model.components.data, expected.components_, atol=1.0e-8)
    assert_allclose(
        nmf_model.transform().data,
        expected.transform(dataset.data),
        atol=1.0e-8,
    )


def test_nmf_transform_fit_transform_and_inverse(nmf_dataset, nmf_model):
    dataset = nmf_dataset
    nmf_model.fit(dataset)

    scores = nmf_model.transform()
    assert scores.shape == (dataset.shape[0], 3)
    assert scores.dims == ["y", "k"]
    assert np.all(scores.data >= -NMF_NONNEGATIVE_TOL)

    scores_from_data = nmf_model.transform(dataset)
    assert_allclose(scores_from_data.data, scores.data, atol=1.0e-8)

    fitted_scores = nmf_model.fit_transform(dataset)
    assert fitted_scores.shape == scores.shape
    assert np.all(fitted_scores.data >= -NMF_NONNEGATIVE_TOL)

    reconstructed = nmf_model.inverse_transform()
    assert reconstructed.shape == dataset.shape
    assert reconstructed.title == dataset.title
    assert reconstructed.units == dataset.units
    assert reconstructed.dims == dataset.dims
    assert_allclose(reconstructed.data, dataset.data, rtol=3.0e-2, atol=1.0e-3)


def test_nmf_masked_data_uses_synthetic_dataset(nmf_dataset):
    dataset = nmf_dataset.copy()
    dataset[:, 4:8] = MASKED

    nmf = NMF(n_components=3, init="nndsvda", max_iter=1000, random_state=123)
    nmf.fit(dataset)
    scores = nmf.transform()

    expected_unmasked_columns = nmf_dataset.shape[1] - 4
    assert nmf._X.shape == (nmf_dataset.shape[0], expected_unmasked_columns)
    assert nmf.X.shape == nmf_dataset.shape
    assert_dataset_equal(nmf.X, dataset)
    assert nmf.components.shape == (3, nmf_dataset.shape[1])
    assert scores.shape == (nmf_dataset.shape[0], 3)
    assert np.all(np.isfinite(scores.data))
    assert np.all(scores.data >= -NMF_NONNEGATIVE_TOL)


def test_nmf_solver_parameter_is_passed_to_sklearn():
    rng = np.random.RandomState(42)
    data = rng.rand(10, 6)

    nmf = NMF(
        n_components=2,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=200,
        random_state=42,
        tol=1e-8,
    )
    nmf.fit(data)
    assert nmf._nmf.solver == "mu"
    assert nmf._nmf.beta_loss == "kullback-leibler"
