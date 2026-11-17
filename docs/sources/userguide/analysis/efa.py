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
# ---

# %% [markdown]
# # Evolving Factor Analysis (EFA)
#
# EFA follows changes in the number of detectable factors along an ordered
# series. Use the observation order, such as acquisition time or elution
# time, and keep a common spectral grid. The forward calculation starts at
# the first row and adds subsequent rows. The backward calculation starts
# at the last row and extends the window toward the first.
#
# ```{=rst}
# See :cite:t:`maeder:1986,maeder:2009` for the method and its interpretation.
# ```
#
# Take every third spectrum of an NH₄Y activation series and mask two
# saturated intervals for all observations before fitting.

# %%
import spectrochempy as scp

spectra = scp.read_omnic("irdata/nh4y-activation.spg")[::3]
spectra.y -= spectra.y[0]
spectra.y.ito("min")
spectra.y.title = "elapsed time"
spectra[:, 1230.0:920.0] = scp.MASKED
spectra[:, 5997.0:5993.0] = scp.MASKED

efa = scp.EFA()
_ = efa.fit(spectra)
forward = efa.f_ev[:, :5]
forward.title = "squared singular value"

# %% [markdown]
# ## Read the forward and backward curves
#
# `f_ev` and `b_ev` contain squared singular values, in descending order
# for each window. Look for curves separating from the lower values as
# each window grows. The two directions give complementary evidence of
# where the detectable rank changes. The curves track rank, and their
# indices do not identify individual chemical species.

# %%
backward = efa.b_ev[:, :5]
backward.title = forward.title
_ = scp.multiplot(
    [forward.T, backward.T],
    labels=["Forward EFA", "Backward EFA"],
    ncol=2,
    method="pen",
    yscale="log",
    sharey=True,
    figsize=(10, 4),
)

# %% [markdown]
# ## Use the curves to initialize a decomposition
#
# Three factors are retained here to illustrate `transform()`. The result
# combines the forward and backward curves into approximate profiles.
# Treat them as starting estimates whose scale and shape depend on the
# retained rank. Use the [MCR-ALS tutorial](mcr_als.rst) for an example
# of refining an EFA initialization with constraints.
#
# The [short gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_efa.rst)
# shows just the forward calculation on the same series.

# %%
efa.n_components = 3
profiles = efa.transform()
profiles.title = "EFA profile (relative scale)"
_ = profiles.T.plot(legend=profiles.k.labels, title="Initial EFA profiles")


# %% [markdown]
# ## Recognize overlapping chromatographic factors
#
# The [simulated chromatogram](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_efa_keller_massart.rst)
# mixes two known spectra with elution profiles that overlap from 5 to
# 7 minutes. When the forward window contains both profiles, its second
# eigenvalue rises well above the third. Starting from the other end
# gives complementary evidence as the window reaches the earlier peak.
#
# For this simulation, the third curve gives an empirical reference for
# the noise level. The graph floors the values at 1e-4 before taking
# the logarithm so that empty modes can be shown. In an experiment, inspect
# with preprocessing and compare with independent information about the
# system. A weak or nearly collinear spectrum can be difficult to detect.
# The `cutoff` setting floors the displayed eigenvalue curves; choose
# `n_components` separately when extracting initial profiles.
