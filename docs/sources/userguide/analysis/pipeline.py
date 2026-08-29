"""
Pipeline.

Pipeline
========

``Pipeline`` composes a small, reproducible sequence of SpectroChemPy
preprocessing transformers and a final transformer or supervised estimator.
It is useful when the same fitted preprocessing must be applied consistently
to calibration and test spectra.

The steps passed to ``Pipeline`` are templates. Calling ``fit()`` clones those
templates, fits the clones, and leaves the original step objects unchanged.
Template steps remain available through ``steps`` and ``named_steps``. Fitted
runtime clones are available after fitting through ``fitted_steps_`` and
``fitted_named_steps_``.
"""

import numpy as np

import spectrochempy as scp

rng = np.random.default_rng(42)
wavenumbers = scp.Coord.linspace(1000.0, 1100.0, 24, title="wavenumber", units="cm^-1")
samples = scp.Coord.arange(12, title="sample")
dataset = scp.NDDataset(
    rng.normal(size=(12, 24)),
    coordset=[samples, wavenumbers],
    dims=["y", "x"],
    units="absorbance",
    title="calibration spectra",
)
target = scp.NDDataset(
    np.linspace(0.1, 1.2, 12),
    coordset=[samples.copy()],
    dims=["y"],
    units="mol/L",
    title="concentration",
)

# %%
# Transformer-final pipelines
# ---------------------------
#
# A transformer-final pipeline ends with a preprocessing transformer or with
# ``PCA``. ``fit_transform(X)`` is equivalent to ``fit(X).transform(X)``.

pipeline = scp.Pipeline(
    [
        ("center", scp.CenterTransformer(dim="y")),
        ("pca", scp.PCA(n_components=3)),
    ]
)
scores = pipeline.fit_transform(dataset)
scores

# %%
# Estimator-final pipelines
# -------------------------
#
# A supervised estimator-final pipeline passes ``y`` only to the final
# estimator. Intermediate preprocessing steps receive only the transformed
# ``NDDataset``.

pipeline = scp.Pipeline(
    [
        ("scale", scp.AutoscaleTransformer(dim="y")),
        ("pls", scp.PLSRegression(n_components=3)),
    ]
)
pipeline.fit(dataset, target)
predicted = pipeline.predict(dataset)
predicted

# %%
# Template and fitted steps
# -------------------------
#
# ``steps``, ``named_steps`` and ``get_params()`` describe the templates.
# Learned runtime state lives on fitted clones.

print(pipeline.named_steps["scale"] is pipeline.fitted_named_steps_["scale"])

# %%
# Nested parameters
# -----------------
#
# Template parameters are visible and editable with ``step__parameter`` names.
# Effective nested changes replace only the modified template step with an
# unfitted clone and preserve untouched template instances. Explicit
# replacement with another compatible instance is always effective, even when
# the new instance has the same constructor parameters.

pipeline.get_params(deep=True)["pls__n_components"]
pipeline.set_params(pls__n_components=2)

# %%
# Supported v1 classes
# --------------------
#
# Intermediate positions accept ``CenterTransformer``, ``AutoscaleTransformer``,
# ``ParetoScaleTransformer``, ``RangeScaleTransformer``,
# ``RobustScaleTransformer``, ``SNVTransformer``, ``NormalizeTransformer``,
# ``MSCTransformer`` and ``LogTransformer``.
#
# Final positions accept those transformers, ``PCA``, ``PLSRegression``,
# ``LSTSQ`` and ``NNLS``.
#
# Version 1 intentionally excludes ``Baseline``, ``SVD``, ``PSD``, ``Filter``,
# functional processing wrappers, ``MCRALS``, ``Optimize``, optional steps,
# branching, caching, persistence guarantees and arbitrary fit-parameter
# routing.
