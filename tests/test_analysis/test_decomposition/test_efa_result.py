# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the EFA result object.
"""

import numpy as np
import pytest

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.efa import EFA


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture()
def efa_dataset():
    """Small synthetic dataset for EFA testing."""
    n_observations = 48
    n_variables = 12

    time = np.linspace(0.0, 1.0, n_observations)
    features = np.linspace(400.0, 700.0, n_variables)

    concentrations = np.column_stack(
        [
            np.exp(-0.5 * ((time - 0.35) / 0.12) ** 2),
            0.8 * np.exp(-0.5 * ((time - 0.68) / 0.14) ** 2),
        ]
    )
    spectra = np.vstack(
        [
            1.0 + 0.3 * np.cos(np.linspace(0.0, np.pi, n_variables)),
            0.7 + 0.4 * np.sin(np.linspace(0.0, np.pi, n_variables)),
        ]
    )
    data = concentrations @ spectra

    import spectrochempy as scp

    return scp.NDDataset(
        data=data,
        coordset=[
            scp.Coord(time, units="minutes", title="time"),
            scp.Coord(features, units="nm", title="wavelength"),
        ],
        title="synthetic EFA mixture",
        units="absorbance",
    )


# ======================================================================================
# EFA result integration tests
# ======================================================================================
class TestEFAResult:
    def test_result_is_analysis_result(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        result = efa.result
        assert isinstance(result, AnalysisResult)
        assert isinstance(result, ResultBase)

    def test_estimator_name(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        assert efa.result.estimator == "EFA"

    def test_outputs_contain_keys(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        result = efa.result
        for name in ("f_ev", "b_ev", "components"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_output_values_match_properties(self, efa_dataset):
        efa = EFA(n_components=2)
        efa.fit(efa_dataset)
        result = efa.result
        np.testing.assert_array_equal(
            result.outputs["f_ev"].data,
            efa.f_ev.data,
        )
        np.testing.assert_array_equal(
            result.outputs["b_ev"].data,
            efa.b_ev.data,
        )
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            efa.components.data,
        )

    def test_diagnostics_is_empty(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        assert efa.result.diagnostics == {}

    def test_repr_contains_expected_fields(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        text = repr(efa.result)
        assert "AnalysisResult" in text
        assert "EFA" in text
        assert "f_ev" in text
        assert "b_ev" in text
        assert "components" in text

    def test_result_is_not_cached(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        assert efa.result is not efa.result, (
            "AnalysisResult is recreated on every access; "
            "change this assertion if caching is added later"
        )

    def test_parameters_contain_expected_keys(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        params = efa.result.parameters
        for name in ("cutoff", "n_components"):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        params = efa.result.parameters
        assert params["cutoff"] is None
        assert params["n_components"] is None

    def test_parameters_match_estimator_config(self, efa_dataset):
        efa = EFA(n_components=2, cutoff=1e-5)
        efa.fit(efa_dataset)
        params = efa.result.parameters
        assert params["n_components"] == 2
        assert params["cutoff"] == 1e-5

    def test_repr_contains_parameters(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        text = repr(efa.result)
        assert "parameters:" in text
        assert "cutoff" in text
        assert "n_components" in text

    def test_raises_before_fit(self):
        efa = EFA()
        with pytest.raises(NotFittedError):
            _ = efa.result

    def test_fit_still_returns_self(self, efa_dataset):
        efa = EFA()
        ret = efa.fit(efa_dataset)
        assert ret is efa

    def test_existing_properties_unchanged(self, efa_dataset):
        efa = EFA()
        efa.fit(efa_dataset)
        assert efa.f_ev.shape == (48, 12)
        assert efa.b_ev.shape == (48, 12)
        assert np.all(np.isfinite(efa.f_ev.data))
        assert np.all(np.isfinite(efa.b_ev.data))
