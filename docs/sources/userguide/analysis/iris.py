# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Integral inversion of spectroscopic data (IRIS)
#
# During a gas adsorption experiment, two surface species can absorb in the
# same spectral region yet respond to pressure over different ranges. A
# series of spectra collected at equilibrium over a range of pressures can
# help separate those contributions. IRIS uses a physical model of the
# pressure response to estimate which contributions fit the entire series.
#
# The result is a map with **wavenumber** on one axis and a **physical
# parameter** on the other. For adsorption, a slice at a given affinity
# shows the spectral features associated with that affinity. A slice at a
# given wavenumber shows how the signal is distributed over affinities.
# The same approach can apply to other experimental variables, such as
# time, when a suitable response model is available.

# %% [markdown]
# ## From measured spectra to a distribution
#
# Let $X(p,\tilde{\nu})$ be the measured signal at pressure $p$ and
# wavenumber $\tilde{\nu}$. The kernel $k(p,q)$ describes how a single
# contribution with parameter $q$ responds to pressure. IRIS estimates the
# distribution $f(q,\tilde{\nu})$ in the model
#
# $$
# X(p,\tilde{\nu}) \approx \int_{q_{\min}}^{q_{\max}}
#     k(p,q)\,f(q,\tilde{\nu})\,\mathrm{d}q.
# $$
#
# The calculation samples $q$ on a grid, incorporates the quadrature weights
# into a matrix $K$, and fits $X \approx Kf$. Each column of $K$ is the
# predicted response of one grid point at the experimental pressures. Each
# column of $f$ is the recovered distribution for one spectral channel.
#
# For adsorption, the Langmuir response takes the form
# $k(p,q)=p\,\exp(-q)/(1+p\,\exp(-q))$ in the gallery convention.
# The pressure at half coverage is numerically $\exp(q)$ in the pressure
# units used to build the kernel. Smaller $q$ corresponds to saturation
# at lower pressure and higher affinity. The figure shows three individual
# responses before quadrature weighting.

# %% [markdown]
# ![Three Langmuir response curves on a logarithmic pressure axis. Smaller q
# reaches half coverage at a lower pressure.](images/iris-langmuir-responses.png)
#
# <details>
# <summary>Show code for the Langmuir curves</summary>
#
# ```python
# import spectrochempy as scp
#
# pressure_values = scp.NDDataset.geomspace(1e-4, 10.0, 200)
# pressure_values.x = scp.Coord(
#     pressure_values.data, title="pressure", units="torr", rounding=False
# )
# responses = []
# labels = []
# for q in (-6, -4, -2):
#     scaled_pressure = pressure_values * scp.exp(-q)
#     coverage = scaled_pressure / (1 + scaled_pressure)
#     coverage.title = "fractional coverage"
#     responses.append(coverage)
#     labels.append(rf"$q = {q}$")
#
# _ = scp.plot_multiple(
#     responses,
#     method="pen",
#     labels=labels,
#     figsize=(7, 3.4),
#     xscale="log",
#     xlim=(1e-4, 10.0),
#     ylim=(0, 1.05),
#     legend="best",
# )
# ```
#
# </details>

# %% [markdown]
# The experimental pressure window determines which responses the data can
# distinguish. If two curves are almost flat or almost identical over that
# window, their contributions are difficult to separate. A finer grid can
# sample the model more closely, but it cannot supply missing experimental
# information. For an adsorption series, check that the pressure coordinate
# is the active observation axis and that its units match the kernel
# convention.

# %% [markdown]
# ## Why regularization matters
#
# Neighboring kernel columns can be very similar. That makes the inverse
# problem ill-conditioned. Small changes in the measured spectra can produce
# large changes in the recovered distribution even when the reconstructed
# spectra look nearly identical.
#
# IRIS estimates nonnegative coefficients and can penalize roughness along
# $q$. The regularization strength $\lambda$ balances residuals against
# roughness. A small $\lambda$ can leave narrow peaks sensitive to noise.
# A large $\lambda$ can merge contributions or leave systematic structure
# in the residuals.
#
# | `reg_par` | What IRIS evaluates |
# | --- | --- |
# | `None` | A nonnegative fit without a roughness penalty |
# | `[-10, 1, 12]` | Twelve logarithmically spaced strengths from $10^{-10}$ to $10^1$ |
# | `[-6, -2]` | An adaptive search between $10^{-6}$ and $10^{-2}$ |
#
# Plot the residual sum of squares against the sharpness measure on
# logarithmic axes to obtain the **L-curve**. Its corner helps identify a
# compromise. Start with a fixed scan to see whether the range covers the
# trade-off, then narrow it for an adaptive search. If the curve has no
# clear corner, compare a few fixed strengths and inspect the reconstructed
# spectra and residuals before interpreting the map.

# %% [markdown]
# ## Read the map alongside the original spectra
#
# The gallery follows CO adsorption on molybdenum supported on alumina. It
# starts with infrared spectra recorded at 19 pressures, selects the CO
# stretching region, builds a Langmuir kernel and estimates an affinity
# distribution.
#
# ![CO adsorption contour map with wavenumber on the horizontal axis and
# adsorption affinity on the vertical axis.](images/iris-co-affinity.png)
#
# *Affinity distribution from the CO gallery example, using the adaptive
# NNLS fit at $\lambda \approx 2.11 \times 10^{-5}$. The contours near
# 2115 cm$^{-1}$ show where the strongest recovered contribution lies in $q$.*
#
# - Inspect `analysis.result.f` as a contour map. A feature extending over
#   several neighboring wavenumbers at similar $q$ can help associate
#   spectral bands with a common pressure response.
# - Compare the reconstructed and measured spectra at representative pressures.
#   Residual structure that persists across the pressure series can indicate
#   that the kernel or preprocessing does not describe some of the signal.
# - Check how the map changes with the grid and $\lambda$. Features that
#   persist across reasonable choices are easier to interpret. The recovered
#   amplitude describes spectral contribution. Conversion to an absolute
#   population requires a calibration or an independent quantitative measure.
#
# ## Try the CO adsorption example
#
# 1. [Install the IRIS plugin](../plugins/iris.rst#installation).
# 2. [Follow the CO adsorption example in the gallery](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_iris_intro.rst).
#    It shows how to attach the pressure coordinate, construct the kernel,
#    compare regularization strengths, and plot the distribution and the
#    reconstructed spectra.
# 3. [Consult the IRIS API reference](../../reference/generated/spectrochempy.iris.IRIS.rst)
#    for estimator parameters, methods and outputs.
#
# [Stelmachowski et al. (2013), *Phys. Chem. Chem. Phys.*](https://doi.org/10.1039/C3CP51146D)
# describe the original 2D IRIS method.
