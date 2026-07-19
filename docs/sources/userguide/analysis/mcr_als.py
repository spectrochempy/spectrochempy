# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.8
# ---

# %% [markdown]
# # MCR-ALS: Multivariate Curve Resolution with Alternating Least Squares
#
# This tutorial covers three progressively complex MCR-ALS workflows:
#
# 1. **Classical MCR-ALS** — a single HPLC-DAD experiment;
# 2. **Constraints** — incorporating prior chemical knowledge;
# 3. **Augmented datasets** — vertical (multi-experiment) and horizontal
#    (multi-technique) data fusion.
#
# Each part answers a practical question the reader is likely to ask.
# The goal is to teach the methodology, not to enumerate every API feature.

# %% [markdown]
# ## Part I — Classical MCR-ALS
#
# ### The bilinear model
#
# MCR-ALS decomposes an experimental data matrix $X$ into the product of two
# physically interpretable factor matrices:
#
# $$ X = C \, S^T + E $$
#
# where:
# - $C$ contains the **concentration profiles** of the pure species
#   (rows → observations (time), columns → components (species));
# - $S^T$ contains the **pure spectra** (rows → components (species),
#   columns → spectral variables (wavelengths));
# - $E$ is the residual.
#
# The ALS algorithm iteratively estimates $C$ from $S^T$ and $S^T$ from $C$
# subject to constraints, until the improvement in the fit falls below a
# tolerance.

# %% [markdown]
# ### 1. Loading and inspecting the data
#
# We use the ALS2004 benchmark dataset from Jaumot et al.,
# *Chemolab*, 76 (2005) 101–110 and *Chemolab*, 140 (2015) 1–12.
# It contains HPLC-DAD measurements of a mixture of four co-eluting species.

# %%
import spectrochempy as scp

datasets = scp.read("matlabdata/als2004dataset.MAT")
for d in datasets:
    print(f"{d.name}: {d.shape}")

# %% [markdown]
# We select the first chromatographic run (`m1`, 51 × 96) which contains the experimental data and the published
# pure-component spectra (`spure`, 4 × 96) which will be used to initialize the MCRALS.

# %%
X = datasets[-1]
St0 = datasets[3]

X.title = "absorbance"
X.set_coordset(None, None)
X.set_coordtitles(y="elution time", x="wavelength")

_ = X.plot(title="experimental spectra")
_ = St0.plot(title="pure-component spectra")

# %% [markdown]
# The 51 rows of `X`are elution times; the 96 columns are wavelengths.
# Overlapping bands cannot be resolved by inspection alone — this is
# where curve resolution is needed.

# %% [markdown]
# ### 2. How many components? — PCA
#
# Principal Component Analysis (PCA) can be used to estimate the number of components.  In particular a scree plot
# of the PCA eigenvalues is often used to provide a quick estimate of the
# effective rank.

# %%
pca = scp.PCA(n_components=8)
pca.fit(X)
_ = pca.plot_scree()

# %% [markdown]
# The scree plot suggests that the effective rank is likely between three and
# four components. The fourth component explains only a small fraction of the
# total variance, so PCA alone does not provide an unambiguous choice.
# The EFA analysis below can give additional information and supports the use of
# four components for the MCR-ALS model.

# %% [markdown]
# ### 3. Where do the components appear? — Evolving Factor Analysis
#
# PCA estimates the overall rank of the dataset but does not indicate where
# individual chemical components are present along the elution axis.
# **Evolving Factor Analysis** (EFA) addresses this question by performing a
# sequence of singular value decompositions (or equivalently PCA) on
# progressively larger submatrices of the data.
#
# In the forward analysis, the decomposition is performed on submatrices
# extending from the beginning of the experiment to each observation in turn.
# In the backward analysis, the same procedure is applied starting from the
# last observation and extending progressively towards the beginning.
#
# The evolution of the significant eigenvalues reflects changes in the local
# rank of the system. It reveals where chemical components enter and leave the
# observation window, providing both an estimate of the number of components
# and approximate bounds for their elution regions.

# %%
efa = scp.EFA()
efa.fit(X)

# Plot forward and backward eigenvalue curves on a log scale
scp.log10(efa.f_ev.clip(1e-4)).T.plot(color="dodgerblue")
_ = scp.log10(efa.b_ev.clip(1e-4)).T.plot(clear=False, color="limegreen")

# %% [markdown]
# Four eigenvalues rise clearly above the noise floor.  The forward curve
# (blue) for each component marks its emergence; the backward curve (green)
# marks its decline.  The zone where the $k$-th forward curve separates
# from the noise is the elution window of component $k$.
#
# From these windows, EFA constructs approximate concentration profiles that
# describe where each component is present along the chromatogram. Although
# these profiles are only estimates, they provide, in this case, an excellent
# initialization for MCR-ALS.

# %%
efa.n_components = 4
C0 = efa.transform()

_ = C0.T.plot()

# %% [markdown]
# EFA provides a physically meaningful initial estimate: each profile
# naturally respects the elution order and peak shape implied by the data.
# It is particularly effective for systems that evolve sequentially
# along the observation axis, such as chromatographic separations where
# components elute in a first-in, first-out order (it was actually initially
# developed for this purpose).

# %% [markdown]
# ### 4. Other initialization methods
#
# Several other strategies exist:
#
# - **SIMPLISMA** — selects purest variables as initial spectra;
# - **Known reference spectra** — when pure-component spectra have been
#   measured independently (e.g. in a spectral library);
# - **Known concentration profiles** — when the elution or kinetic model is
#   known a priori.
#
# In this tutorial we will use successively the pure spectra `spure` as the initial $S^T$ (stored in `St0`) and the EFA initialization as the initial $C$ (stored in `C0`).
# and compare the two approaches.


# %% [markdown]
# ### 5. Running the optimisation
#
# The default configuration already applies non-negativity to both $C$ and
# $S^T$, and unimodality to $C$ — sensible defaults for chromatographic
# data.
#
# Let's first run the MCR-ALS algorithm with the pure spectra as initial guess for $S^T$.

# %%
mcr_s = scp.MCRALS(log_level="INFO")
_ = mcr_s.fit(X, St0)

# %% [markdown]
# ### 6. Inspecting the solution
#
# The estimated profiles preserve the coordinate metadata of the input data.

# %%
C = mcr_s.C
St = mcr_s.St

print(f"C: {C.shape}")
print(f"St: {St.shape}")

_ = C.T.plot(title="Concentration profiles")
_ = St.plot(title="Pure spectra")

# %% [markdown]
# The reconstruction quality can be assessed with `plot_merit`:

# %%
_ = mcr_s.plot_merit(offset=5)

# %% [markdown]
# Numerical diagnostics confirm convergence:

# %%
print(f"Residual std: {mcr_s.result.diagnostics['residual_std']:.6f}")
print(f"Converged: {mcr_s.result.converged}")
print(f"Iterations: {mcr_s.result.n_iter}")

# %% [markdown]
# Now let's run the MCR-ALS algorithm with the EFA initialization for $C$ and the pure spectra as initial guess for $S^T$ and compare the two approaches.
# %%0
mcr_c = scp.MCRALS(log_level="INFO")
_ = mcr_c.fit(X, C0)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 6))
mcr_c.C.T.plot(ax=axes[0, 0], title="C (EFA init)")
mcr_s.C.T.plot(ax=axes[0, 1], title="C (Pure spectra init)")
mcr_c.St.plot(ax=axes[1, 0], title="$S^T$ (EFA init)")
mcr_s.St.plot(ax=axes[1, 1], title="$S^T$ (Pure spectra init)")
mcr_c.plot_merit(ax=axes[2, 0], offset=5)
mcr_s.plot_merit(ax=axes[2, 1], offset=5)
plt.tight_layout()
plt.show()

# %% [markdown]
# The EFA initializations converges less rapidly but provides very
# similar reconstruction, as shown by the comparable residuals in the merit
# plots. This indicates that the bilinear model describes the experimental
# data consistently, regardless of whether the initial estimate is supplied
# for $C$ or for $S^T$.
#
# Note, however, that the amplitudes of the resolved profiles differ
# substantially. This reflects the intrinsic scale ambiguity of MCR-ALS:
# multiplying one column of $C$ by a constant and dividing the corresponding
# row of $S^T$ by the same constant leaves the product $CS^T$ unchanged.
# Consequently, the absolute magnitudes of the concentration and spectral
# profiles should not be compared directly unless a common normalization
# convention is imposed.
#
# Differences are also observed in the shapes of the resolved profiles.
# These arise because EFA provides only an approximate initial estimate and
# because the spectra of several components overlap strongly. Such spectral
# correlation makes the decomposition less well conditioned, allowing slightly
# different distributions of variance between components while producing
# nearly identical reconstructions of the experimental data. This behaviour is
# typical of bilinear inverse problems and illustrates why chemically
# meaningful constraints are often essential to obtain a unique and
# interpretable solution.
#

# %% [markdown]
# ## Part II — Constraints
#
# The true power of MCR-ALS lies in the ability to inject prior chemical
# knowledge through constraints.  SpectroChemPy provides a **declarative
# constraint API**: each constraint is a simple object that describes *what*
# is known.
#
# The most important practical question is: **which constraint should I use
# for my scientific problem?**

# %% [markdown]
# ### Quick reference
#
# The constraints are imported from `spectrochempy.analysis.constraints`:
#
# | Constraint          | Profile | Purpose                                                    |
# |---------------------|---------|------------------------------------------------------------|
# | `NonNegative`       | C, St   | Non-negativity of concentrations or spectra                |
# | `Unimodal`          | C, St   | Single maximum per component (chromatography, kinetics)    |
# | `Closure`           | C       | Constant sum of selected components (mass balance)         |
# | `Monotonic`         | C       | Monotonic increase or decrease (kinetic or thermodynamic profiles)          |
# | `ModelProfile`      | C, St   | Hard model from a user-supplied callable (kinetic model)   |
# | `ComponentPresence` | C       | Component absent in certain blocks (augmented data)        |
# | `Trilinear`         | C       | Rank-1 constraint across blocks (augmented data)           |
#
# All constraints accept optional `blocks=` (see below) and `components=` parameters to
# restrict their scope.

# %% [markdown]
# #### Non-negativity
#
# Non-negativity is the most widely used constraint in MCR-ALS because both
# chemical concentrations and absorbance spectra are inherently non-negative.
# For this reason, the default MCR-ALS configuration in SpectroChemPy applies
# this constraint automatically to both concentration and spectral profiles.

# To illustrate its effect, we first remove all constraints and solve the
# problem using unconstrained least squares

# %%
mcr_nc = scp.MCRALS(log_level="INFO", constraints=[])
mcr_nc.fit(X, C0)
_ = mcr_nc.C.T.plot(title="Non-constrained concentration profiles")

# %% [markdown]
# The unconstrained least-squares solution contains negative concentrations and
# oscillatory profiles.
#
# ##### How is non-negativity enforced?
# There are two common strategies for enforcing non-negativity during the ALS
# iterations.
#
# (i) The default configuration follows the classical MCR-ALS strategy introduced
#  by Tauler and co-workers: each ALS half-step first computes an unconstrained
#  least-squares solution:
#
# $$ C_{LS} = X(S^T)^+ $$
#
# after which the `NonNegative` constraint then projects this solution onto the
# non-negative domain by replacing negative values with zero:
#
# $$ C_{constrained} = \max(C_{LS}, 0) $$
#
# This approach is computationally inexpensive. Here we demonstrate this
# for concentrations:
# %%
from spectrochempy.analysis import constraints as ct

mcr_nn = scp.MCRALS(log_level="INFO", constraints=[ct.NonNegative("C")])
mcr_nn.fit(X, C0)
_ = mcr_nn.C.T.plot(title="Non-negative concentration profiles (projection)")

# %% [markdown]
#
# (ii) Alternatively, the least-squares problem itself can be solved under the
# non-negativity constraint,
#
# $$ C_{\mathrm{NNLS}} =
#    \arg\min_{C \ge 0} \|X-CS^T\|_F^2 $$
#
# using a dedicated non-negative least-squares (NNLS) solver. In this case,
# positivity is enforced during the least-squares optimization rather than by
# projecting an unconstrained solution afterwards. The resulting estimate is
# the exact non-negative least-squares solution, at the expense of a higher
# computational cost.
#
# The choice of solver is independent of the constraint definition. The
# `NonNegative` constraint specifies that the solution must be non-negative,
# whereas the ALS solver determines whether this constraint is enforced by
# projection (`lstsq`) or directly during the optimization (`nnls` or
# `pnnls`).
#
# Here we demonstrate the `nnls` solver for concentrations:
# %%
from spectrochempy.analysis import constraints as ct

mcr_nnl = scp.MCRALS(log_level="INFO", solver_C="nnls", constraints=[])

mcr_nnl.fit(X, C0)
_ = mcr_nnl.C.T.plot(title="Non-negative concentration profiles (NNSL solver)")

# %% [markdown]
#
# Here we obtain non-negative concentration profiles without projection. Note however that
# some profiles are not unimodal -- this point will be addressed in a following section.
#
# Below we show how to apply non-negativity constraints particular profiles:
# %%

# All concentration profiles (default)
_ = ct.NonNegative("C")

# All spectral profiles (default)
_ = ct.NonNegative("St")

# Selected concentration components
_ = ct.NonNegative("C", components=[0, 1])

# Selected spectral components
_ = ct.NonNegative("St", components=[0])

# Selected blocks of an augmented dataset (see below)
_ = ct.NonNegative("C", blocks=[1, 2])
_ = ct.NonNegative("St", blocks=[0])  # e.g. UV spectra only

# Combine component and block selections
_ = ct.NonNegative("C", components=[0], blocks=[2])

# %% [markdown]
# Constraints are supplied as a list when constructing the MCR-ALS estimator.

# %%
mcr = scp.MCRALS(
    constraints=[
        ct.NonNegative("C"),
        ct.NonNegative(
            "St", blocks=[0]
        ),  # e.g. only the first block has non-negative spectra
    ],
    log_level="INFO",
)

# %% [markdown]
# #### `Unimodality`
#
# In chromatography or kinetics each concentration profile typically has a single maximum.

# %%
_ = ct.Unimodal("C", mod="strict")  # strict single maximum (default)
_ = ct.Unimodal("C", mod="smooth")  # allows a flat-topped region
_ = ct.Unimodal("C", tolerance=1.0)  # no allowance for local violations

# %% [markdown]
# Here we illustrate

mcr_nnl_u = scp.MCRALS(
    log_level="INFO", solver_C="nnls", constraints=[ct.Unimodal("C")]
)

mcr_nnl_u.fit(X, C0)
_ = mcr_nnl_u.C.T.plot(title="NNSL solver + unimodality")

# %% [markdown]
# Unimodality can also be applied to spectral profiles when thy consist in a single band:

# %%
_ = ct.Unimodal("St", components=[1])  # only the second component

# %% [markdown]
# #### `Closure`
#
# Useful when the total concentration of the mixture is known.

# %%
_ = ct.Closure("C")  # rows sum to 1.0
_ = ct.Closure("C", target=100.0)  # custom constant sum
_ = ct.Closure("C", target=[1.0, 0.9, 0.8])  # per-row targets

# %% [markdown]
# #### `Monotonic`
#
# For kinetic profiles that only increase or decrease.

# %%
_ = ct.Monotonic("C", "increasing", components=[0])
_ = ct.Monotonic("C", "decreasing", components=[1])

# %% [markdown]
# #### `ModelProfile`
#
# When a profile shape is known from a physical model (kinetic rate law,
# peak-shape function), the ALS estimate is replaced by the model output
# at each iteration.

# %%
# def kinetic_model(C_current):
#     return C_constrained
#
# ct.ModelProfile("C", model=kinetic_model)

# %% [markdown]
# #### `ComponentPresence`
#
# For augmented (multi-block) data: specify which components are present
# in each concentration block.

# %%
# ct.ComponentPresence("C", presence=[
#     [True, True, True, True],
#     [True, True, True, False],
# ])

# %% [markdown]
# #### `Trilinear` — the rank-one assumption
#
# Trilinearity is the only constraint we demonstrate in full because it
# addresses a recurring question in multi-experiment MCR-ALS.
#
# When the same mixture is measured repeatedly under identical conditions
# (e.g. several HPLC injections), each component should have the **same
# concentration profile shape** across runs, differing only by a
# per-experiment **amplitude**.  Mathematically, the concentration profiles
# of each component form a rank‑1 matrix across experiments:
#
# $$ \begin{bmatrix} c_1 & c_2 & \dots & c_N \end{bmatrix}
#    = a \, b^T $$
#
# where $a$ is the common shape and $b$ contains the amplitudes.
# The `Trilinear` constraint enforces this by projecting the
# per-component, multi-block profiles onto their best rank‑1 approximation
# at every ALS iteration.
#
# Repeated chromatographic runs of the same mixture naturally satisfy this
# assumption.  The constraint is applied during *vertical augmentation*,
# which we cover in the next section.
#
# A practical note: the default MCR-ALS configuration already applies
# `NonNegative` and `Unimodal` to the concentration profiles, so explicitly
# adding these constraints will barely change the ALS2004 solution.  They
# are the first constraints to consider for any new dataset; the more
# specialised constraints above become relevant when the default model
# needs refinement.

# %% [markdown]
# ## Part III — Augmented datasets
#
# MCR-ALS can simultaneously analyse multiple data matrices.
# SpectroChemPy supports two augmentation modes, each corresponding to a
# different scientific scenario.

# %% [markdown]
# ### A. Vertical augmentation — multiple experiments, common spectra
#
# In **vertical (column-wise) augmentation** several experiments are stacked
# row-wise.  Each experiment has its own concentration profiles, but the
# pure spectra $S^T$ are common to all.
#
# $$ \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_N \end{bmatrix}
#    = \begin{bmatrix} C_1 \\ C_2 \\ \vdots \\ C_N \end{bmatrix}
#      S^T + E $$
#
# The ALS2004 dataset contains the `MATRIX` variable: four successive
# HPLC-DAD runs concatenated into a single 204 × 96 matrix.

# %%
X2 = datasets[1]
X2.title = "absorbance"
X2.set_coordset(None, None)
X2.set_coordtitles(y="elution time", x="wavelength")

X_blocks = [X2[i * 51 : (i + 1) * 51].copy() for i in range(4)]
for i, b in enumerate(X_blocks):
    b.name = f"run {i + 1}"
    b.title = "absorbance"
    print(f"{b.name}: {b.shape}")

# %% [markdown]
# The blocks share the same spectral variables (96 wavelengths) but have
# their own elution-time axes.  When all blocks have identical dimensions
# (51 × 96), we must specify `augmentation="vertical"` explicitly.

# %%
mcr_v = scp.MCRALS(log_level="INFO")
_ = mcr_v.fit(X_blocks, St0, augmentation="vertical")

# %% [markdown]
# The concentration matrix is the row-wise concatenation of the individual
# $C_k$ blocks.  The `C_blocks` property restores the per-experiment view:

# %%
print(f"C (concatenated): {mcr_v.C.shape}")
print(f"St (common): {mcr_v.St.shape}")

for i, cb in enumerate(mcr_v.C_blocks):
    print(f"C_blocks[{i}]: {cb.shape}")

# %%
_ = scp.multiplot(
    [cb.T for cb in mcr_v.C_blocks],
    nrow=2,
    ncol=2,
    suptitle="Concentration profiles — vertical augmentation",
)

# %% [markdown]
# #### Trilinearity
#
#
# Repeated chromatographic runs of the same mixture should produce
# concentration profiles with identical shapes, differing only in scale.
# We can enforce this with the `Trilinear` constraint.

# %%
mcr_vt = scp.MCRALS(
    constraints=[ct.NonNegative("C"), ct.NonNegative("St"), ct.Trilinear("C")],
    log_level="INFO",
)
_ = mcr_vt.fit(X_blocks, St0, augmentation="vertical")

# %% [markdown]
# For the ALS2004 benchmark, the four runs are already highly consistent profiles
# (cross-block correlations > 0.998), so the constraint does not
# substantially change the fit:

# %%
print(
    f"Without trilinearity — residual std: {mcr_v.result.diagnostics['residual_std']:.4f}"
)
print(
    f"With trilinearity    — residual std: {mcr_vt.result.diagnostics['residual_std']:.4f}"
)

_ = scp.multiplot(
    [cb.T for cb in mcr_vt.C_blocks],
    nrow=2,
    ncol=2,
    suptitle="Concentration profiles — trilinear constraint",
)

# %% [markdown]
# The value of the `Trilinear` constraint becomes apparent with noisier
# data or when runs exhibit slight instrumental drift.  In those cases,
# forcing a common shape removes artefactual differences and produces
# chemically more interpretable profiles.

# %% [markdown]
# ### B. Horizontal augmentation — common observations, different spectroscopies
#
# In **horizontal (row-wise) augmentation** several data matrices share
# the observation axis (same number of rows) but have different spectral
# variables.  The concentration profiles $C$ are common, while each
# technique contributes its own spectral block $S_i^T$.
#
# $$ \begin{bmatrix} X_1 & X_2 & \dots & X_N \end{bmatrix}
#    = C \, \begin{bmatrix} S_1^T & S_2^T & \dots & S_N^T \end{bmatrix}
#      + E $$
#
# This is useful when the same sample is measured with complementary
# spectroscopic techniques — the joint analysis forces a single
# concentration profile to explain all techniques simultaneously.

# %% [markdown]
# #### Loading the DNA thermal denaturation dataset
#
# The dataset contains UV absorbance and CD (circular dichroism) spectra
# recorded during a DNA melting experiment: 24 temperature steps from
# 21 °C to 90 °C, 101 wavelengths from 230 nm to 330 nm.

# %%
dna = scp.read_matlab("matlabdata/dna_data.mat", merge=False)

temp_coord = scp.Coord(dna[1].data.flatten(), title="temperature", units="°C")
wave_coord = scp.Coord(dna[0].data.flatten(), title="wavelength", units="nm")

X_uv = scp.NDDataset(dna[3].data, name="UV", title="absorbance")
X_uv.add_coordset(x=wave_coord, y=temp_coord)

X_cd = scp.NDDataset(dna[2].data, name="CD", title="ellipticity")
X_cd.add_coordset(x=wave_coord, y=temp_coord)

print(f"UV: {X_uv.shape}")
print(f"CD: {X_cd.shape}")

# %%

_ = scp.multiplot(
    (X_uv, X_cd),
    nrow=1,
    ncol=2,
    labels=("UV spectra", "CD spectra"),
    suptitle="DNA melting",
)

# %% [markdown]
# #### Building a deterministic initial guess
#
# We use an EFA-based initialization on the combined UV+CD data.

# %%
# concatenate the two datasets
combined = scp.concatenate(
    X_uv, X_cd
)  # a warning is raised because we concatenate absorbances and ellipticities

# fit an EFA
efa = scp.EFA().fit(combined)

# and plot forward and backward curves to deternamlibne
scp.log10(efa.f_ev.clip(1e-5)).T.plot(color="dodgerblue")
_ = scp.log10(efa.b_ev.clip(1e-5)).T.plot(clear=False, color="limegreen")

# %% [markdown]
# The plot above show that only two components are identified (the matrix is exactly of rank 2).

# %%
efa.n_components = 2
C0 = efa.transform()
_ = C0.T.plot()

# %% [markdown]
# #### Choosing the right constraints
#
# UV absorption spectra are non-negative but CD spectra are **signed** (positive and negative
# bands), so global non-negativity on $S^T$ would be incorrect.
#
# We therefore apply `NonNegative` only to the **UV spectral block**
# (block 0):

# %%
mcr_h = scp.MCRALS(
    constraints=[
        ct.NonNegative("C"),
        ct.NonNegative("St", blocks=[0]),
        ct.Closure("C"),
    ],
    log_level="INFO",
    tol_reconstruction_error=1e-3,
)
_ = mcr_h.fit([X_uv, X_cd], C0, augmentation="horizontal")

# %% [markdown]
# The concentration profiles $C$ carry the temperature coordinate.
# The concatenated spectral matrix $S^T$ is split into technique-specific
# blocks via `St_blocks`, each preserving its own wavelength coordinate.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
_ = mcr_h.C.T.plot(ax=axes[0], title="$C$")
_ = mcr_h.St_blocks[0].plot(ax=axes[1], title="$S^T_1$ (UV)")
_ = mcr_h.St_blocks[1].plot(ax=axes[2], title="$S^T_2$ (CD)")
plt.tight_layout()


# %% [markdown]
# The resolved profiles separate native (folded) and denatured (unfolded)
# DNA.  The UV block shows two distinct absorption spectra; the CD block
# shows the characteristic signed bands of the folded structure declining
# as temperature increases.

# %% [markdown]
# ## Summary — choosing the right workflow
#
# The three parts of this tutorial map directly to practical questions:
#
# ```
# One experiment
#         │
#         ▼
#  Classical MCR-ALS
#  (Part I)
#
# Repeated experiments
# (common spectra)
#         │
#         ▼
# Vertical augmentation
# (Part III-A)
#         │
#         ├── runs are consistent    →  no trilinearity needed
#         └── runs need shape alignment → add ct.Trilinear("C")
#
# Multiple techniques
# (common concentrations)
#         │
#         ▼
# Horizontal augmentation
# (Part III-B)
#         │
#         └── per-block constraints    →  ct.NonNegative("St", blocks=[...])
# ```
#
