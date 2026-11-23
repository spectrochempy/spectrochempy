# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Augmented MCR-ALS with fixed reference concentrations
=====================================================

In this example, the spectra of independent mixtures are augmented with the
spectra of reference samples of known composition. The concentrations of the
reference samples are fixed at each ALS iteration with a ``ModelProfile``
constraint, which replaces the deprecated ``hardConc`` and ``getConc``
parameters. The resolved concentrations of the mixtures are then expressed in
the units of the reference concentrations.
"""

import numpy as np

import spectrochempy as scp
from spectrochempy.analysis import constraints as ct

# %%
# Synthetic data
# --------------
# Three pure components are built from Gaussian bands.
x = scp.Coord.linspace(1000.0, 1800.0, 401, units="cm^-1", title="wavenumber")
pure = scp.stack(
    scp.gaussian(x, ampl=1.0, pos=1200.0, width=80.0, normalized=False)
    + scp.gaussian(x, ampl=0.4, pos=1500.0, width=60.0, normalized=False),
    scp.gaussian(x, ampl=0.8, pos=1350.0, width=100.0, normalized=False),
    scp.gaussian(x, ampl=0.6, pos=1550.0, width=70.0, normalized=False)
    + scp.gaussian(x, ampl=0.3, pos=1250.0, width=50.0, normalized=False),
)

# %%
# Twelve mixtures of random composition and three reference samples, each
# containing a single component at a known concentration, are measured with
# some noise.
c_mix = 2.0 * scp.random((12, 3), seed=7)
c_ref = np.diag([0.5, 1.0, 1.5])

X_mix = scp.dot(c_mix, pure) + 0.005 * scp.normal(size=(12, 401), seed=1)
X_mix.x = x
X_mix.title = "absorbance"
X_ref = scp.dot(scp.NDDataset(c_ref), pure) + 0.005 * scp.normal(size=(3, 401), seed=2)
X_ref.x = x
X_ref.title = "absorbance"
_ = X_mix.plot()


# %%
# Constraints
# -----------
# The mixtures are independent samples, not a time series: the unimodality
# of the concentration profiles, applied by default, is not relevant here. The
# constraints are therefore given explicitly. The model function receives the
# current concentrations of the selected block (here the reference block,
# ``blocks=[1]``) and returns the known concentrations.
def reference_concentrations(C):
    return c_ref


constraints = [
    ct.NonNegative("C"),
    ct.NonNegative("St"),
    ct.ModelProfile("C", model=reference_concentrations, blocks=[1]),
]

# %%
# Vertical augmentation
# ---------------------
# Both blocks share the spectral axis, so the datasets are stacked vertically.
# The measured reference spectra are used as the initial guess of the pure
# spectra.
mcr = scp.MCRALS(constraints=constraints, max_iter=200)
_ = mcr.fit([X_mix, X_ref], X_ref)
print(f"iterations: {mcr.result.diagnostics['n_iter']}")
print(f"reference concentrations:\n{mcr.C_blocks[1].data}")

# %%
# For comparison, the same fit without the reference constraint gives
# concentrations in the arbitrary scale of the initial guess.
mcr_free = scp.MCRALS(constraints=constraints[:2], max_iter=200)
_ = mcr_free.fit([X_mix, X_ref], X_ref)
for name, model in (("fixed references", mcr), ("free", mcr_free)):
    error = np.abs(model.C_blocks[0].data - c_mix.data).max()
    print(f"{name}: max. concentration error of the mixtures = {error:.3f}")

# %%
# Resolved concentrations of the mixtures against the true ones, all
# components together:
_ = scp.plot_parity(
    scp.NDDataset(c_mix.data.ravel()),
    scp.NDDataset(mcr.C_blocks[0].data.ravel()),
    label="mixture concentrations",
)

# %%
# Resolved pure spectra:
_ = mcr.St.plot()

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
