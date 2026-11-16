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
# ## Work through the CO adsorption series
#
# [Install the IRIS plugin](../plugins/iris.rst#installation) to run this tutorial.
# The dataset contains 19 infrared spectra measured during CO adsorption.
# Use the recorded pressures as the active observation coordinate and keep
# the acquisition times as a secondary coordinate. Select the CO stretching
# region using its wavenumber limits.

# %%
import spectrochempy as scp

spectra = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
pressures = [
    0.003,
    0.004,
    0.009,
    0.014,
    0.021,
    0.026,
    0.036,
    0.051,
    0.093,
    0.150,
    0.203,
    0.300,
    0.404,
    0.503,
    0.602,
    0.702,
    0.801,
    0.905,
    1.004,
]
pressure = scp.Coord(pressures, title="pressure", units="torr")
spectra.y = [spectra.y.copy(), pressure]
spectra.y.select(2)
spectra = spectra[:, 2250.0:1950.0]
_ = spectra.plot_contourf(colorbar=True)

# %% [markdown]
# ### Build the kernel and fit without regularization
#
# The Langmuir kernel uses 50 grid points between $q=-8$ and $q=-1$.
# The fit without a penalty gives a baseline for judging what the penalty
# changes in the recovered distribution.

# %%
kernel = spectra.iris.kernel_matrix(kernel_type="langmuir", q=[-8, -1, 50])
unregularized = scp.iris.IRIS(reg_par=None)
_ = unregularized.fit(spectra, kernel)
_ = unregularized.result.f[0].plot_contour()

# %% [markdown]
# ### Scan regularization strengths
#
# Scan a broad range and read the L-curve. For this dataset the bend is
# near $10^{-4}$ on this grid. Use the corresponding slice of the distribution
# to see how the roughness penalty changes the map.

# %%
scan = scp.iris.IRIS(reg_par=[-10, 1, 12])
_ = scan.fit(spectra, kernel)
_ = scan.plotlcurve(title="Fixed scan of regularization strengths")
_ = scan.result.f[6].plot_contour()

# %% [markdown]
# ### Refine the search and inspect the reconstruction
#
# Search within a narrower range around the bend. The default NNLS solver
# reports the selected strength and its index when logging is set to INFO.
# For this fit the search selects about $2.11 \times 10^{-5}$ at index 4.
# Use the index reported by your fit to select the matching distribution
# and reconstruction.

# %%
adaptive = scp.iris.IRIS(reg_par=[-6, -2], log_level="INFO")
_ = adaptive.fit(spectra, kernel)
_ = adaptive.plotlcurve(title="Adaptive search")
selected_index = 4
_ = adaptive.result.f[selected_index].plot_contour()
_ = adaptive.plot_merit(index=selected_index)

# %% [markdown]
# The contours near 2115 cm$^{-1}$ show where the strongest recovered
# contribution lies in $q$. Bands extending across neighboring wavenumbers
# at similar $q$ may share a pressure response. Check the reconstructed
# spectra and residuals against the measurements. Residual structure that
# persists across pressures can indicate a mismatch in the kernel or
# preprocessing. Check that the map features persist across reasonable
# choices of grid and $\lambda$. The recovered amplitude describes a
# spectral contribution. Converting it to an absolute population requires
# calibration or an independent quantitative measure.
#
# The [short CO gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_iris_intro.rst)
# shows one fit at a fixed strength. See the
# [IRIS API reference](../../reference/generated/spectrochempy.iris.IRIS.rst)
# for estimator parameters, methods and outputs.
#
# ```{=rst}
# :cite:t:`stelmachowski:2013` describe the original 2D IRIS method.
# ```
