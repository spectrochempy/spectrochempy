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
# # Integral-equation Regularized Inversion for Spectroscopy (IRIS)
#
# A series of spectra recorded as pressure, time, or another experimental
# variable changes can help distinguish contributions whose bands overlap.
# IRIS fits the whole series using a response law shared by the contributions.
# Each contribution responds to the experimental variable according to its
# own physical parameter, such as an adsorption affinity or a rate constant.
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
# Let $X(p,\epsilon)$ be the measured signal at experimental setting $p$
# and spectral channel $\epsilon$. The kernel $k(p,q)$ describes the
# response of a contribution with parameter $q$. For discrete contributions,
# the signal is a sum of spectra weighted by their respective responses.
# For a continuous distribution $f(q,\epsilon)$, the sum becomes an integral.
#
# $$
# X(p,\epsilon) \approx \int_{q_{\min}}^{q_{\max}}
#     k(p,q)\,f(q,\epsilon)\,\mathrm{d}q.
# $$
#
# The calculation samples $q$ on a grid, incorporates the quadrature weights
# into a matrix $K$, and fits $X \approx Kf$. Each column of $K$ is the
# response of one grid point at the experimental settings, weighted for
# quadrature. Each column of $f$ is the recovered distribution for one
# spectral channel.
#
# Built-in kernels cover first-order reactant decay and product growth,
# adsorption equilibrium, diffusion uptake, and pulsed-field-gradient NMR
# (`stejskal-tanner`). You can also supply a callable for your own response law.
# Check that the chosen law describes the contributions in the spectral
# region you fit. For example, a monotonic kernel cannot represent a transient
# intermediate whose signal first grows and then decays.
#
# ### Example kernel for adsorption
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
# A peak's width along $q$ can reflect a spread of physical parameters,
# departures from the assumed response law, or the resolution of the
# inversion. Try neighboring values of $\lambda$ and different grid
# spacings, and use independent evidence to interpret persistent features.
#
# | `reg_par` | What IRIS evaluates |
# | --- | --- |
# | `None` | A nonnegative fit without a roughness penalty |
# | `[-10, 1, 12]` | Twelve logarithmically spaced strengths from $10^{-10}$ to $10^1$ |
# | `[-6, -4]` | An adaptive search between $10^{-6}$ and $10^{-4}$ |
#
# Plot the sharpness measure (SM) against the residual sum of squares (RSS)
# on logarithmic axes to obtain the **L-curve**. Small $\lambda$ can fit noise
# and produce high sharpness at nearly constant RSS. At large $\lambda$, the
# distribution varies little with $\lambda$ while the residuals increase.
# The corner helps identify a compromise. See
# `hansen:1992`{.interpreted-text role="cite:t"} for the L-curve criterion and
# `bauer:2011`{.interpreted-text role="cite:t"} for a comparison with other
# parameter selection methods.
#
# ![Schematic L-curve showing the compromise between residuals and roughness.](images/iris-lcurve-schematic.png)
#
# Start with a fixed scan to see whether the range covers the
# trade-off, then narrow it for an adaptive search. If the curve has no
# clear corner, compare a few fixed strengths and inspect the reconstructed
# spectra and residuals before interpreting the map.

# %% [markdown]
# ## Work through the CO adsorption series
#
# [Install the IRIS plugin](../plugins/iris.rst#installation) to run this tutorial.
# The dataset contains 19 infrared difference spectra of CO measured at
# increasing pressures on a sulfided CoMo/Al$_2$O$_3$ catalyst
# `oliviero:2021`{.interpreted-text role="cite:p"}.
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
_ = spectra.plot(colorbar=True)

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
_ = unregularized.result.f[0].plot_contour(equal_aspect=False)

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
_ = scan.result.f[6].plot_contour(equal_aspect=False)

# %% [markdown]
# ### Refine the search and inspect the reconstruction
#
# Search within a narrower range around the bend. The default NNLS solver
# reports the selected strength and its index when logging is set to INFO.
# For these bounds the search selects about $7.45 \times 10^{-5}$ at index 9.
# Use the index reported by your fit to select the matching distribution
# and reconstruction.

# %%
adaptive = scp.iris.IRIS(reg_par=[-6, -4], log_level="INFO")
_ = adaptive.fit(spectra, kernel)
_ = adaptive.plotlcurve(title="Adaptive search")

# %% [markdown]
# The result contains the evaluated strengths (`lambdas`), the two series
# plotted on the L-curve (`RSS` and `SM`), and the fit configuration
# (`parameters`). Use the same index to inspect the distribution and both
# diagnostics for a particular strength.

# %%
result = adaptive.result
selected_index = 9
print(
    "λ, RSS, SM:",
    result.lambdas.data[selected_index],
    result.RSS[selected_index],
    result.SM[selected_index],
)
print(result.parameters)
_ = result.f[selected_index].plot_contour(equal_aspect=False)
_ = adaptive.plot_merit(index=selected_index)

# %% [markdown]
# ### Interpret the distribution and residuals
#
# Features at lower $q$ reach half coverage at lower pressures. In this
# catalyst, bands near 2110 cm$^{-1}$ are associated with CO on unpromoted
# molybdenum sulfide edges. Bands below about 2090 cm$^{-1}$ include
# contributions from cobalt-promoted sites, as discussed in the study cited
# above. A band that shifts with pressure can leave a curved feature in the
# map, so use the original spectra when interpreting the contours.
#
# Check the reconstructed spectra and residuals against the measurements.
# Residual structure that
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
