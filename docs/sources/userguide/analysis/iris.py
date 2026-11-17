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
# In many chemical systems, the response of a species to an external
# variable $p$ (time, pressure, field gradient…) is governed, to a first
# approximation, by a single internal thermodynamic or kinetic parameter
# $q$, through a known two-variable law: the normalized concentration of
# species $i$ is $c_i(p) = k(p, q_i)$. Real systems rarely contain a single
# species: they combine contributions that follow the same law with
# different values $q_i$. Since spectroscopic signals are proportional to
# concentration, a series of spectra measured as a function of $p$ reads
#
# $$
# X(p, \epsilon) = \sum_i k(p, q_i)\, s_i(\epsilon),
# $$
#
# where $\epsilon$ is the spectroscopic channel (energy, wavenumber,
# wavelength, chemical shift…) and $s_i(\epsilon)$ the spectrum of species
# $i$. When the $q_i$ are numerous or spread continuously (heterogeneous
# environments, for example), the sum becomes an integral:
#
# $$
# X(p, \epsilon) = \int_{q_{\min}}^{q_{\max}} k(p, q)\, f(q, \epsilon)\,
#     \mathrm{d}q,
# $$
#
# where $f(q, \epsilon)$ is the spectral distribution of the parameter $q$
# and the law $k(p, q)$ is the kernel of the integral. Recovering $f$
# requires inverting this Fredholm integral equation of the first kind, an
# ill-posed problem that IRIS solves with Tikhonov regularization
# `stelmachowski:2013`{.interpreted-text role="cite:p"}. The result is a map
# with the spectroscopic channel on one axis and the parameter $q$ on the
# other: a slice at a given $q$ shows the spectrum of the species with that
# parameter, and a slice at a given channel shows how the signal is
# distributed over $q$.
#
# Note that this model, and therefore any inversion of it, assumes that
# every contribution follows the same monotonic law. Indeed, without a
# priori or external knowledge, the spectra only reveal the net change: when
# some species appear while others disappear in the same spectral region,
# only the balance is accessible, much like judging inflow and outflow from
# the water level alone. Intermediates that rise and fall cannot be
# described either. IRIS further constrains the distribution to be
# non-negative.

# %% [markdown]
# ## Response laws
#
# SpectroChemPy provides kernels for the following laws:
#
# * **First-order kinetics** (`"reactant-first-order"`,
#   `"product-first-order"`): $p$ is the time and $q = \ln k$, the logarithm
#   of the rate constant.
# * **Adsorption equilibrium** (`"langmuir"`, `"ca"` for the condensation
#   approximation): $p$ is the pressure and $q$ the reduced adsorption free
#   energy $\Delta_{ads}G^0/RT$.
# * **Diffusion uptake** (`"diffusion"`): $p$ is the time and $q$ the
#   diffusion rate constant $\tau^{-1}$.
# * **Pulsed-field-gradient NMR** (`"stejskal-tanner"`): $p$ is the
#   $b$-value and $q = \log D$, the logarithm of the diffusivity.
#
# Other processes, not directly implemented, can also be analyzed by
# supplying the law $k(p, q)$ as a Python function, for example:
#
# * relaxation and decay processes (luminescence decays, NMR $T_1$/$T_2$
#   relaxation, dielectric relaxation), with $q = \ln\tau$;
# * acid–base or complexation equilibria followed by spectroscopic titration,
#   with $p$ the pH and $q$ the p$K_a$, when the selected bands belong to a
#   single form;
# * redox equilibria in spectroelectrochemistry, with $p$ the applied
#   potential and $q$ the formal potential, under the same condition;
# * thermally activated processes such as temperature-programmed desorption,
#   with $p$ the temperature and $q$ the activation energy.
#
# ### Example kernel for adsorption
#
# For adsorption, the Langmuir response takes the form
# $k(p,q)=p\,\exp(-q)/(1+p\,\exp(-q))$ in the gallery convention.
# The pressure at half coverage is numerically $\exp(q)$ in the pressure
# units used to build the kernel. Smaller $q$ corresponds to saturation
# at lower pressure and higher affinity. The figure shows three individual
# responses.

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
# window, their contributions are difficult to separate. A finer grid of $q$
# can sample the model more closely, but it cannot supply missing
# experimental information. For an adsorption series, check that the
# pressure coordinate is the active observation axis and that its units
# match the kernel convention.

# %% [markdown]
# ## Why regularization matters
#
# Recovering $f$ from the spectra is an ill-posed problem, for three reasons
# that the regularization must reconcile:
#
# 1. **Instability.** The kernel $k(p, q)$ generally varies smoothly with
#    $q$: over the experimental range of $p$, the responses for close values
#    of $q$ are nearly indistinguishable. Small differences in the spectra,
#    such as noise or baseline drifts, can therefore produce large changes
#    in $f$, including spurious peaks, even when the reconstructed spectra
#    look nearly identical.
# 2. **Intrinsic widths.** Real distributions are not sums of delta
#    functions: heterogeneous environments or species spread over a range of
#    $q$ and give peaks with an intrinsic width.
# 3. **Non-ideality.** A species whose response deviates from the ideal law
#    $k(p, q)$, because of a concentration-dependent energy for instance,
#    also appears as a broadened distribution rather than as a single value
#    of $q$.
#
# IRIS therefore penalizes the roughness of $f$ along $q$, after sampling $q$
# on a grid. The regularization strength $\lambda$ balances residuals
# against roughness: a small $\lambda$ leaves the solution sensitive to
# noise, while a large $\lambda$ merges contributions or leaves systematic
# structure in the residuals. Note that the regularization itself broadens
# peaks. The width of a peak along $q$ can thus reflect a genuine
# distribution, the non-ideality of a single species, or the limited
# resolution of the inversion, and the inversion alone cannot separate these
# contributions (see the note on peak width and shape at the end of this
# tutorial).

# %% [markdown]
# ## Choosing $\lambda$ with the L-curve
#
# Several criteria exist to choose the regularization strength, for example:
#
# * the **discrepancy principle**, which requires the residuals to match the
#   noise level, and therefore a good estimate of that noise;
# * **generalized cross-validation (GCV)**, which selects the $\lambda$ that
#   best predicts left-out data;
# * **Bayesian or maximum-likelihood approaches**, which treat $\lambda$ as a
#   statistical parameter;
# * the **L-curve**, which looks for the best compromise between residuals
#   and roughness.
#
# SpectroChemPy currently implements the L-curve, which is probably the most
# intuitive. For each $\lambda$, IRIS computes the residual sum of squares
# (RSS) and the sharpness of the distribution (SM), and plots
# $\log \mathrm{SM}$ against $\log \mathrm{RSS}$. The curve usually has an
# L shape (see the figure below):
#
# * at small $\lambda$, the vertical branch: residuals hardly decrease
#   further while the distribution becomes increasingly rough, a sign that
#   the fit follows the noise;
# * at large $\lambda$, the horizontal branch: the distribution is smooth,
#   but residuals grow because real features are smoothed out;
# * the **corner**, where the curvature is maximal, marks the best
#   compromise.
#
# ![Schematic L-curve: the vertical branch at small λ corresponds to
# under-smoothed solutions, the horizontal branch at large λ to over-smoothed
# solutions, and the corner to the best
# compromise.](images/iris-lcurve-schematic.png)
#
# The `reg_par` parameter of `IRIS` controls which values of $\lambda$ are
# evaluated:
#
# | `reg_par` | What IRIS evaluates |
# | --- | --- |
# | `None` | A nonnegative fit without a roughness penalty |
# | `[-10, 1, 12]` | Twelve logarithmically spaced strengths from $10^{-10}$ to $10^1$ |
# | `[-6, -4]` | An adaptive search of the corner between $10^{-6}$ and $10^{-4}$ |
#
# The adaptive search locates the point of maximum curvature by a
# golden-section search `cultrera:2020`{.interpreted-text role="cite:p"}.
# Start with a fixed scan to check that the range covers both branches, then
# narrow it for an adaptive search. If the curve has no clear corner, compare
# a few fixed strengths and inspect the reconstructed spectra and residuals
# before interpreting the map.
#
# See `hansen:1992`{.interpreted-text role="cite:t"} for the L-curve
# criterion, and `bauer:2011`{.interpreted-text role="cite:t"} for a review
# and comparison of the main parameter choice methods.

# %% [markdown]
# ## Work through the CO adsorption series
#
# CO is a common probe of catalyst surfaces: it binds (adsorbs) to specific
# surface sites, and each type of site gives a CO species with its own
# infrared band and its own binding strength. Strong sites fill at low
# pressure, weak sites only at higher pressure.
#
# [Install the IRIS plugin](../plugins/iris.rst#installation) to run this
# tutorial. The dataset contains 19 infrared difference spectra of CO
# measured at increasing pressures on a sulfided Mo/Al$_2$O$_3$ catalyst
# `oliviero:2021`{.interpreted-text role="cite:p"}. Some bands overlap, in
# particular in the broad massif of the sulfide phase at low wavenumbers,
# where the spectra alone hardly show how many species contribute and how
# strongly each binds CO. IRIS helps separate these contributions according
# to their binding strength $q$.
#
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
# ### Interpret the distribution
#
# Each spot of the map is a CO species: its wavenumber gives its band, and
# its $q$ its binding strength (the more negative, the stronger). From the
# weakest to the strongest bound `oliviero:2021`{.interpreted-text role="cite:p"}:
#
# * physisorbed CO (~2145 cm$^{-1}$, $q \approx -1.5$) and CO on alumina OH
#   groups (~2155 cm$^{-1}$, $q \approx -3$);
# * CO on the S-edges of the molybdenum sulfide (~2070 cm$^{-1}$,
#   $q \approx -4.5$);
# * CO on the M-edges of the molybdenum sulfide (~2110 cm$^{-1}$,
#   $q \approx -5$);
# * CO on the exposed aluminum ions of the support, from weaker sites
#   (~2175 cm$^{-1}$, $q \approx -3.5$) to stronger ones (~2200 cm$^{-1}$
#   and above, $q \approx -5.5$).
#
# Note that the weaker aluminum sites give an arc rather than a spot: their
# band shifts with pressure because the sites are not all identical and
# because adsorbed CO molecules interact with each other as the surface
# fills up (see the note below). On cobalt-promoted catalysts, additional
# bands of the promoted sulfide phase appear below about 2090 cm$^{-1}$ at
# more negative values of $q$.
#
# Finally, compare the reconstructed spectra with the measured ones
# (`plot_merit`). Above about 0.1 Torr, the residuals are only a few percent
# of the signal. At the lowest pressures, where coverage is low, the
# deviations are much larger (about 30 % of a weak signal) and vary
# systematically with pressure rather than randomly. They may reflect a
# departure from the Langmuir model at low coverage or a larger uncertainty
# on the lowest pressures; their detailed analysis is beyond the scope of
# this tutorial. The recovered amplitude describes a spectral contribution.
# Converting it to an absolute population requires calibration or an
# independent quantitative measure.

# %% [markdown]
# ### Note on the interpretation of peak width and shape
#
# As discussed above, the width and shape of a peak along $q$ can have three
# origins: a genuine distribution of sites or species, the non-ideality of a
# single species, and the limited resolution of the inversion
# (regularization, noise, finite range of $p$). The inversion alone cannot
# separate them, but a few checks help:
#
# * invert synthetic data for one ideal species, with the same values of
#   $p$, a similar noise level and the same $\lambda$: only the width in
#   excess of this reference is meaningful;
# * repeat the fit for neighboring values of $\lambda$ and grids: features
#   that change with these choices come from the inversion, not from the
#   sample;
# * look at the shape: a spot elongated or curved along $q$, like the arc of
#   the aluminum sites, means that the band shifts as the surface fills up;
# * use chemical knowledge and complementary experiments (other probe
#   molecules, temperatures or samples) to decide between heterogeneity and
#   non-ideality.
#
# The [short CO gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_iris_intro.rst)
# shows one fit at a fixed strength. See the
# [IRIS API reference](../../reference/generated/spectrochempy.iris.IRIS.rst)
# for estimator parameters, methods and outputs.
