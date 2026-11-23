Reweighted baseline models
==========================

The :class:`~spectrochempy.Baseline` processor offers two explicitly selected
alternatives to AsLS: ``model="arpls"`` (asymmetrically reweighted penalized
least squares) and ``model="aspls"`` (adaptive smoothness penalized least
squares). Existing models and defaults are unchanged. Both implementations
use NumPy and SciPy; installing pybaselines is not required.

For a measured dataset ``X``, choose a model and inspect its estimated baseline
before interpreting the corrected peak areas:

.. code-block:: python

    import spectrochempy as scp

    processor = scp.Baseline(
        model="arpls", lamb=1e6, tol=1e-3, max_iter=100
    )
    processor.fit(X)
    baseline = processor.baseline
    corrected = processor.corrected

    adaptive = scp.Baseline(
        model="aspls", lamb=1e7, aspls_asymmetry=0.5,
        tol=1e-3, max_iter=100
    )
    adaptive.fit(X)
    adaptive_corrected = adaptive.corrected

These parameter values illustrate the syntax; they are not universal presets.
The existing helpers also accept these model names:

.. code-block:: python

    baseline = X.get_baseline(model="arpls", lamb=1e6, max_iter=100)
    corrected = X.basc(model="aspls", lamb=1e7, max_iter=100)

Input and parameter contract
----------------------------

Both models accept finite real, unmasked 1D spectra and rowwise 2D datasets
with at least three regularly spaced points along the last axis. Increasing
and decreasing coordinates are supported. Coordinates, units and shape are
preserved; corrected outputs retain source metadata and append a baseline
correction history entry with the effective parameters.

Masked data and irregular grids raise ``ValueError``. Neither silently
dropping masked points nor treating their spacing as uniform is a valid
replacement. Multivariate fitting, LLS and breakpoints are also rejected for
these two models. They do not change the behavior of the existing models.

``lamb`` must be finite and positive. It penalizes second differences in
sample indices, so its appropriate value depends on the sampling density,
baseline curvature and signal. Increasing it generally makes the baseline
stiffer. A changed point count can require a different value even over the
same physical interval.

``tol`` must satisfy ``0 < tol < 1`` and measures relative changes in the
weights. ``max_iter`` is the maximum number of linear solves (at least one).
A ``RuntimeWarning`` reports failure to satisfy the convergence criterion,
including insufficient negative residuals to estimate the noise. A finite
baseline may still be returned, but it must be inspected before use.
For asPLS, floating-point roundoff can change the stopping iteration and
whether the weight tolerance is reached. Check the stability of the
estimated baseline as well as the convergence warning.

``aspls_asymmetry`` is the positive logistic coefficient used only by asPLS.
The default 0.5 follows the pybaselines variant; the original publication
uses 2. This is distinct from the existing AsLS ``asymmetry`` parameter,
which is unused by arPLS and asPLS.

Scientific limits
-----------------

Neither method can universally distinguish a broad band from a curved
background. Broad weak bands and negative peaks can be absorbed into the
estimated baseline, even when weight convergence is reached. Check the
baseline, vary the parameters and validate areas against suitable standards
or known additions. Do not choose a method solely because the corrected
spectrum appears flatter. These models do not perform automatic method
selection or shared fitting across acquisitions.

References
-----------

* Baek et al., *Analyst* 140 (2015), 250–257,
  `arPLS <https://doi.org/10.1039/C4AN01061B>`_.
* Zhang et al., *Spectroscopy Letters* 53 (2020), 222–233,
  `asPLS <https://doi.org/10.1080/00387010.2020.1730908>`_.
* The weighting conventions follow
  `pybaselines 1.2.1 <https://pybaselines.readthedocs.io/en/stable/algorithms/whittaker.html>`_.
  Its BSD notice is retained in the numerical module.
