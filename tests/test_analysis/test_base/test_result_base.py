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
