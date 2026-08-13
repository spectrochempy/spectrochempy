# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Optimize provenance and derived-identity policy tests.

`Optimize` inherits the analysis source snapshot from `AnalysisConfigurable`.
Its fitted outputs and residual datasets must now follow the accepted
analysis-output metadata policy for source-domain derived results.
"""

from datetime import UTC
from datetime import datetime

import numpy as np

import spectrochempy as scp


def test_optimize_fitted_and_components_preserve_source_author(
    synthetic_two_peak_dataset, optimize_script
):
    ds = synthetic_two_peak_dataset
    ds.author = "optimize_author"
    ds.origin = "optimize_origin"
    ds.name = "optimize_source"
    ds.acquisition_date = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    ds.meta.project = "curvefit"
    data_before = ds.data.copy()

    opt = scp.Optimize(script=optimize_script).fit(ds)
    result = opt.result

    assert result.fitted.author == "optimize_author"
    assert result.fitted.origin == "optimize_origin"
    assert result.fitted.name == "optimize_source_Optimize.fitted_data"
    assert result.fitted.title == "fitted data"
    assert result.fitted.description == (
        "Fitted data from Optimize fit of optimize_source."
    )
    assert result.fitted.units == ds.units
    assert result.fitted.filename is None
    assert result.fitted.meta.project == "curvefit"
    assert result.fitted.meta is not ds.meta
    assert result.fitted.acquisition_date == ds.acquisition_date
    assert result.fitted.history[-1].endswith(
        "Created fitted data with Optimize from optimize_source."
    )
    assert opt.predict().units == ds.units
    assert result.components.author == "optimize_author"

    # Numeric no-regression: residuals still equal observed minus fitted.
    expected = ds - result.fitted
    np.testing.assert_allclose(
        np.ma.asarray(result.residuals.masked_data),
        np.ma.asarray(expected.masked_data),
    )
    assert result.residuals.units == ds.units
    assert result.residuals.author == "optimize_author"
    assert result.residuals.origin == "optimize_origin"
    assert result.residuals.name == "optimize_source_Optimize.residuals"
    assert result.residuals.title == "residuals"
    assert result.residuals.description == (
        "Residuals from Optimize fit of optimize_source."
    )
    assert result.residuals.filename is None
    assert result.residuals.meta.project == "curvefit"
    assert result.residuals.meta is not ds.meta
    assert result.residuals.acquisition_date == ds.acquisition_date
    assert result.residuals.history[-1].endswith(
        "Created residuals with Optimize from optimize_source."
    )

    # Input must not be mutated by the fit.
    assert np.array_equal(ds.data, data_before)
    assert ds.author == "optimize_author"


def test_optimize_fitted_and_residuals_preserve_none_units(
    synthetic_two_peak_dataset, optimize_script
):
    ds = synthetic_two_peak_dataset.to(None, force=True)

    opt = scp.Optimize(script=optimize_script).fit(ds)
    result = opt.result

    assert result.fitted.units is None
    assert opt.predict().units is None
    assert result.residuals.units is None


def test_optimize_fitted_and_residuals_restore_public_geometry_and_mask(
    synthetic_two_peak_dataset, optimize_script
):
    ds = synthetic_two_peak_dataset.copy()
    ds[40] = scp.MASKED

    opt = scp.Optimize(script=optimize_script).fit(ds)
    observed = opt.X
    result = opt.result

    for output in (result.fitted, result.residuals):
        assert output.shape == observed.shape
        assert output.dims == observed.dims
        np.testing.assert_array_equal(output.mask, observed.mask)
        assert output.coordset is not observed.coordset
        np.testing.assert_allclose(output.x.data, observed.x.data)

    result.fitted.mask[0, 0] = True
    assert not observed.mask[0, 0]


def test_optimize_predict_accepts_readonly_source_metadata(
    synthetic_two_peak_dataset, optimize_script
):
    ds = synthetic_two_peak_dataset.copy()
    ds.name = "readonly_source"
    ds.meta.project = "nmr-relax"
    ds.meta.processing = {"reader": "topspin"}
    ds.meta.readonly = True

    opt = scp.Optimize(script=optimize_script).fit(ds)

    prediction = opt.predict()
    residuals = opt.result.residuals

    assert prediction.name == "readonly_source_Optimize.fitted_data"
    assert prediction.meta.project == "nmr-relax"
    assert prediction.meta.processing["reader"] == "topspin"
    assert prediction.meta.readonly is False
    prediction.meta.processing["reader"] = "changed"
    assert ds.meta.processing["reader"] == "topspin"

    assert residuals.meta.project == "nmr-relax"
    assert residuals.meta.processing["reader"] == "topspin"
    assert residuals.meta.readonly is False
    residuals.meta.processing["reader"] = "residuals-changed"
    assert ds.meta.processing["reader"] == "topspin"
