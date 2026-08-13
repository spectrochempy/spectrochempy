# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Processing a saturation-recovery relaxation series
==================================================

This example processes a pseudo-2D series of 1D spectra acquired with a
variable recovery delay and fits a simple saturation-recovery model to the
dominant resonance.

The bundled Bruker dataset stores the delays in ``vdlist`` and the pulse
program indicates a CP/MAS saturation-recovery experiment. The example keeps a
modest public scope:

- process the 1D FIDs with explicit SpectroChemPy operations;
- extract a signal trace from the dominant processed resonance;
- fit a simple two-parameter recovery model to that trace.

It does not claim replay of vendor processing, nor exact equivalence with the
TopSpin fitting tools bundled alongside the dataset.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
# Import API
# ----------
import spectrochempy as scp

# short version of the unit registry
U = scp.ur

# %%
# Import a pseudo-2D delay series
# -------------------------------
# Define the folder containing the Bruker experiment.
datadir = scp.preferences.datadir
nmrdir = datadir / "nmrdata" / "bruker" / "tests" / "nmr"

dataset = scp.nmr.read(nmrdir / "relax" / "100" / "ser", use_list="vdlist")

# %%
# Analysing the data
# ---------------------
# Print dataset summary
print(dataset)

# %%
# Plot the processed spectra
# --------------------------
# The `vdlist` delays become the secondary coordinate of the pseudo-2D series.
ds = dataset.em(lb=15 * U.Hz)
ds = ds.fft()
ds = ds.pk(phc0=-145 * U.deg, phc1=0 * U.deg)
_ = ds.plot(xlim=(100, -50))

# %%
# Build a signal trace from the dominant resonance
# -----------------------------------------------
# The strongest processed peak in this series sits around 20–22 ppm.
# We integrate a narrow ppm window around that resonance for each delay.
signal = ds[:, 20.0:45.0].simpson()
_ = signal.plot(marker="^", ls=":")
signal.real

# %%
# Fit a model
# -----------
# Keep the gallery output focused on the fitted result rather than optimizer
# progress chatter, which otherwise renders as an empty "Out:" block.
fitter = scp.Optimize(log_level="WARNING", method="leastsq")


# %%
# Define the model to fit
def T1_model(t, I0, T1):  # no underscore in parameters names.
    # Simple saturation-recovery model.
    import numpy as np

    I = I0 * (1 - np.exp(-t / T1))
    return I


# %%
# Add the model to the fitter usermodels as it it not a built-in model
fitter.usermodels = {"T1_model": T1_model}

# %%
# Define the parameter variables using a script
# (parameter: value, low_bound,  high_bound)
# no underscore in parameters names.
fitter.script = """
MODEL: T1
shape: T1_model
  $ I0:  1000.0, 1, none
  $ T1:  2.0,    0.1, none
"""

# %%
# Perform the fit
_ = fitter.fit(signal)

# %%
som = fitter.predict()
print(som)

# %%
# Plot the measured recovery points and the fitted curve separately so the
# experimental series remains a true scatter plot.
ax = signal.plot_scatter(
    color="tab:blue",
    marker="o",
    markersize=5,
    label="measured signal",
    title="Saturation-recovery fit of the dominant resonance",
)
_ = som.plot(clear=False, color="tab:orange", lw=1.8, label="fitted curve")
_ = ax.legend()

# %%
# This ends the example ! The following line can be removed or commented
# when the example is run as a notebook (`.ipynb`).

# scp.show()
