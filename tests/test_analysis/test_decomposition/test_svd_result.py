# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the SVD result object contract prototype.
"""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.svd import SVD
from tests.test_analysis.result_test_helpers import assert_fit_returns_self
from tests.test_analysis.result_test_helpers import assert_result_basics
from tests.test_analysis.result_test_helpers import assert_result_raises_before_fit


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture()
def low_rank_dataset():
    y = scp.Coord.arange(4, title="sample")
    x = scp.Coord.arange(5, title="feature")
    data = np.zeros((4, 5))
    data[0, 0] = 5.0
    data[1, 1] = 3.0
    return scp.NDDataset(
        data,
        coordset=[y, x],
        units="absorbance",
        title="synthetic low-rank matrix",
    )


# ======================================================================================
# SVD result integration tests
# ======================================================================================
class TestSVDResult:
    def test_result_is_analysis_result(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = assert_result_basics(svd, AnalysisResult, "SVD")
        assert isinstance(result, ResultBase)

    # ----------------------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------------------
    def test_outputs_contain_keys(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = svd.result
        for name in ("U", "s", "VT"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_output_values_match_properties(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = svd.result
        np.testing.assert_array_equal(result.outputs["U"], svd.U)
        np.testing.assert_array_equal(result.outputs["s"], svd.s)
        np.testing.assert_array_equal(result.outputs["VT"], svd.VT)

    def test_output_types_preserved(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = svd.result
        assert isinstance(result.outputs["U"], np.ndarray)
        assert isinstance(result.outputs["s"], np.ndarray)
        assert isinstance(result.outputs["VT"], np.ndarray)

    def test_output_shapes_default(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = svd.result
        assert result.outputs["U"].shape == (4, 4)
        assert result.outputs["s"].shape == (4,)
        assert result.outputs["VT"].shape == (4, 5)

    # ----------------------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------------------
    def test_diagnostics_contain_keys(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = svd.result
        for name in (
            "singular_values",
            "explained_variance",
            "explained_variance_ratio",
        ):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_diagnostic_values_match_properties(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        result = svd.result
        np.testing.assert_array_equal(
            result.diagnostics["singular_values"].data,
            svd.singular_values.data,
        )
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance"].data,
            svd.explained_variance.data,
        )
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance_ratio"].data,
            svd.explained_variance_ratio.data,
        )

    # ----------------------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------------------
    def test_parameters_contain_expected_keys(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        params = svd.result.parameters
        for name in ("full_matrices", "compute_uv"):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values_default(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        params = svd.result.parameters
        assert params["full_matrices"] is False
        assert params["compute_uv"] is True

    def test_parameters_match_estimator_config(self, low_rank_dataset):
        svd = SVD(full_matrices=True, compute_uv=True)
        svd.fit(low_rank_dataset)
        params = svd.result.parameters
        assert params["full_matrices"] is True
        assert params["compute_uv"] is True

    # ----------------------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------------------
    def test_repr_contains_expected_fields(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        text = repr(svd.result)
        assert "AnalysisResult" in text
        assert "SVD" in text
        assert "U" in text
        assert "s" in text
        assert "VT" in text
        assert "singular_values" in text
        assert "explained_variance" in text
        assert "full_matrices" in text

    # ----------------------------------------------------------------------------------
    # Pre-fit guard
    # ----------------------------------------------------------------------------------
    def test_raises_before_fit(self):
        assert_result_raises_before_fit(SVD(), NotFittedError)

    # ----------------------------------------------------------------------------------
    # Existing behaviour preserved
    # ----------------------------------------------------------------------------------
    def test_fit_still_returns_self(self, low_rank_dataset):
        assert_fit_returns_self(SVD(), low_rank_dataset)

    def test_existing_properties_unchanged(self, low_rank_dataset):
        svd = SVD()
        svd.fit(low_rank_dataset)
        assert svd.U.shape == (4, 4)
        assert svd.s.shape == (4,)
        assert svd.VT.shape == (4, 5)
        assert svd.sv.shape == (4,)
        np.testing.assert_allclose(svd.s, [5.0, 3.0, 0.0, 0.0])

    def test_compute_uv_false(self, low_rank_dataset):
        svd = SVD(compute_uv=False)
        svd.fit(low_rank_dataset)
        # The _outfit normalisation fix (PR #1210) ensures _outfit is always
        # a (U, s, VT) tuple, so all properties and diagnostics work
        # correctly even when compute_uv=False.
        result = svd.result
        assert isinstance(result, AnalysisResult)
        assert result.estimator == "SVD"
        # outputs — U and VT are None, s is correct
        assert result.outputs["U"] is None
        np.testing.assert_array_equal(
            result.outputs["s"],
            [5.0, 3.0, 0.0, 0.0],
        )
        assert result.outputs["VT"] is None
        # diagnostics are fully populated
        for name in (
            "singular_values",
            "explained_variance",
            "explained_variance_ratio",
        ):
            assert name in result.diagnostics, f"{name} missing from diagnostics"
        # diagnostics values match properties
        np.testing.assert_array_equal(
            result.diagnostics["singular_values"].data,
            svd.singular_values.data,
        )
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance"].data,
            svd.explained_variance.data,
        )
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance_ratio"].data,
            svd.explained_variance_ratio.data,
        )
        # params still reflect user configuration
        assert result.parameters["compute_uv"] is False
        assert result.parameters["full_matrices"] is False
