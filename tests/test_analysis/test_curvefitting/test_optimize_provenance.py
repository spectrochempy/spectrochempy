# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Optimize no-regression provenance tests.

``Optimize`` inherits the analysis provenance snapshot from
``AnalysisConfigurable``.  Its fitted and components outputs are produced
through the ``transform`` path and must preserve the scientific source
``author``, while the numeric fit results must be unchanged.

``residuals`` is a pure-arithmetic output (observed - fitted) and keeps the
runtime user/host author; aligning it with the scientific source is deferred
to the later multi-source policy PR (PR 2).
"""

import numpy as np

import spectrochempy as scp


def test_optimize_fitted_and_components_preserve_source_author(
    synthetic_two_peak_dataset, optimize_script
):
    ds = synthetic_two_peak_dataset
    ds.author = "optimize_author"
    data_before = ds.data.copy()

    opt = scp.Optimize(script=optimize_script).fit(ds)
    result = opt.result

    assert result.fitted.author == "optimize_author"
    assert result.components.author == "optimize_author"

    # Numeric no-regression: residuals still equal observed minus fitted.
    expected = ds - result.fitted
    np.testing.assert_allclose(
        np.ma.asarray(result.residuals.masked_data),
        np.ma.asarray(expected.masked_data),
    )
    assert result.residuals.units == ds.units

    # Input must not be mutated by the fit.
    assert np.array_equal(ds.data, data_before)
    assert ds.author == "optimize_author"
