# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the shared Result object infrastructure.
"""

import numpy as np
import pytest

from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import FitResult
from spectrochempy.analysis._base._result import ResultBase


class TestResultBase:
    def test_minimal_creation(self):
        result = ResultBase(estimator="TestEstimator")
        assert result.estimator == "TestEstimator"
        assert result.outputs == {}
        assert result.diagnostics == {}
        assert result.parameters == {}

    def test_with_outputs_and_diagnostics(self):
        result = ResultBase(
            estimator="Test",
            outputs={"a": np.ones(3), "b": np.ones((2, 3))},
            diagnostics={"d": np.array([1.0, 2.0])},
        )
        assert "a" in result.outputs
        assert "b" in result.outputs
        assert "d" in result.diagnostics

    def test_output_attribute_access(self):
        scores = np.ones((2, 3))
        concentrations = object()
        spectra = object()
        result = ResultBase(
            estimator="Test",
            outputs={"scores": scores, "C": concentrations, "St": spectra},
        )
        assert result.scores is result.outputs["scores"]
        assert result.C is result.outputs["C"]
        assert result.St is result.outputs["St"]

    def test_diagnostic_attribute_access(self):
        variance = np.array([2.0, 1.0])
        result = ResultBase(
            estimator="Test",
            diagnostics={"explained_variance": variance},
        )
        assert result.explained_variance is result.diagnostics["explained_variance"]

    def test_missing_attribute_raises_attribute_error(self):
        result = ResultBase(estimator="Test")
        with pytest.raises(
            AttributeError,
            match="ResultBase object has no attribute 'missing'",
        ):
            _ = result.missing

    def test_existing_attributes_are_not_shadowed(self):
        result = ResultBase(
            estimator="Test",
            parameters={"alpha": 1},
            outputs={
                "outputs": "shadowed outputs",
                "diagnostics": "shadowed diagnostics",
                "parameters": "shadowed parameters",
                "estimator": "shadowed estimator",
            },
        )
        assert result.outputs["outputs"] == "shadowed outputs"
        assert result.diagnostics == {}
        assert result.parameters == {"alpha": 1}
        assert result.estimator == "Test"

    def test_output_wins_over_diagnostic(self):
        output = object()
        diagnostic = object()
        result = ResultBase(
            estimator="Test",
            outputs={"shared": output},
            diagnostics={"shared": diagnostic},
        )
        assert result.shared is output

    def test_parameters_are_not_exposed_as_attributes(self):
        result = ResultBase(estimator="Test", parameters={"alpha": 1})
        with pytest.raises(AttributeError):
            _ = result.alpha

    def test_dir_includes_outputs_and_diagnostics_but_not_parameters(self):
        result = ResultBase(
            estimator="Test",
            parameters={"alpha": 1},
            outputs={"scores": object(), "not-valid": object()},
            diagnostics={"explained_variance": object()},
        )
        names = dir(result)
        assert "scores" in names
        assert "explained_variance" in names
        assert "alpha" not in names
        assert "not-valid" not in names

    def test_repr(self):
        result = ResultBase(
            estimator="Test",
            outputs={"x": np.ones((3, 4))},
            diagnostics={"y": np.ones(2)},
        )
        text = repr(result)
        assert "ResultBase" in text
        assert "Test" in text
        assert "x" in text
        assert "(3, 4)" in text
        assert "y" in text


class TestAnalysisResult:
    def test_creation(self):
        result = AnalysisResult(estimator="PCA")
        assert isinstance(result, AnalysisResult)
        assert isinstance(result, ResultBase)

    def test_repr(self):
        result = AnalysisResult(
            estimator="PCA",
            outputs={"scores": np.ones((6, 5))},
            diagnostics={"explained_variance": np.ones(5)},
        )
        text = repr(result)
        assert "AnalysisResult" in text
        assert "PCA" in text


class TestFitResult:
    def test_creation(self):
        result = FitResult(estimator="Optimize")
        assert isinstance(result, FitResult)
        assert isinstance(result, ResultBase)

    def test_inherits_attribute_access(self):
        fitted = object()
        result = FitResult(estimator="Optimize", outputs={"fitted": fitted})
        assert result.fitted is fitted

    def test_covariance_property(self):
        covariance = np.eye(2)
        result = FitResult(estimator="Optimize", covariance=covariance)
        np.testing.assert_array_equal(result.covariance, covariance)

    def test_variance_property_derived_from_covariance(self):
        covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
        result = FitResult(estimator="Optimize", covariance=covariance)
        np.testing.assert_array_equal(result.variance, np.array([4.0, 9.0]))
        assert result.variance.flags.writeable is False

    def test_stderr_property_derived_from_variance(self):
        covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
        result = FitResult(estimator="Optimize", covariance=covariance)
        np.testing.assert_array_equal(result.stderr, np.array([2.0, 3.0]))
        assert result.stderr.flags.writeable is False


class TestResultBaseHTMLRepr:
    """Behavioral tests for the HTML representation of ResultBase."""

    def test_repr_html_returns_valid_html(self):
        result = ResultBase(estimator="PCA")
        html = result._repr_html_()
        assert html.startswith('<div class="scp-output">')
        assert "PCA" in html

    def test_repr_html_has_collapsible_sections(self):
        result = ResultBase(
            estimator="Test",
            parameters={"n_components": 5, "tol": 1e-6},
            outputs={"scores": np.ones((10, 5))},
            diagnostics={"explained_variance": np.array([0.9, 0.1])},
        )
        html = result._repr_html_()
        assert "<details" in html
        assert "Parameters (2)" in html
        assert "Outputs" in html
        assert "Diagnostics" in html
        assert "tol" in html
        assert "scores" in html

    def test_repr_html_empty_sections_omitted(self):
        result = ResultBase(estimator="Test")
        html = result._repr_html_()
        assert "Parameters" not in html
        assert "Outputs" not in html
        assert "Diagnostics" not in html

    def test_repr_html_embeds_nddataset_repr(self):
        from spectrochempy import NDDataset

        scores = NDDataset(np.ones((10, 5)))
        result = ResultBase(
            estimator="PCA",
            outputs={"scores": scores},
        )
        html = result._repr_html_()
        assert "NDDataset" in html
        assert "float64" in html

    def test_repr_html_heading_shows_estimator(self):
        result = ResultBase(estimator="MyEstimator")
        heading = result._repr_html_()
        assert "MyEstimator" in heading

    def test_repr_html_analysis_result(self):
        result = AnalysisResult(
            estimator="PCA",
            parameters={"n_components": 3},
            outputs={"scores": np.ones((10, 3))},
        )
        html = result._repr_html_()
        assert "PCA" in html
        assert "AnalysisResult" in html

    def test_repr_html_fit_result(self):
        result = FitResult(
            estimator="Optimize",
            outputs={"fitted": np.ones((10,))},
        )
        html = result._repr_html_()
        assert "Optimize" in html
        assert "FitResult" in html

    def test_repr_html_output_count_in_header(self):
        result = ResultBase(
            estimator="Test",
            outputs={"a": np.ones(3), "b": np.ones((2, 3)), "c": np.ones(5)},
        )
        html = result._repr_html_()
        assert "Outputs (3)" in html
