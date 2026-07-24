# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Two-dimensional NMR processing
==============================
End-to-end 2D NMR processing from raw SER data to a 2D frequency-domain spectrum.

This example complements :ref:`sphx_glr_plugins_spectrochempy-nmr_examples_processing_nmr_plot_processing_nmr.py`:

* :mod:`plot_processing_nmr.py` starts from a TopSpin ``2rr`` dataset that is
  already processed.
* This example starts from the raw TopSpin ``ser`` file and performs the main
  processing steps inside SpectroChemPy.

The pipeline is intentionally split in two stages:

1. F2 (direct dimension): apodization -> FFT -> auto-phase
2. F1 (indirect dimension): zero-filling -> apodization -> FFT

It illustrates:

* reading raw 2D SER data;
* automatic handling of Bruker encodings such as ``STATES-TPPI``;
* quaternion-aware 2D FFT processing;
* how to obtain a spectrum comparable to the processed ``2rr`` reference.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
# Import API
# ----------
import spectrochempy as scp

# %%
# Read raw 2D SER data
# --------------------
# The raw SER file contains time-domain data together with the acquisition
# metadata needed to decode the hypercomplex encoding in F1.

nmrdir = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr"
ser = scp.nmr.read(nmrdir / "topspin_2d", expno=1, remove_digital_filter=True)

# %%
# Print dataset summary — both dimensions are in time domain
ser

# %%
# Plot the raw time-domain data (quaternion representation)
_ = ser.plot_map()

# %%
# For comparison, load the TopSpin-processed reference spectrum.
# This is similar to the dataset used in ``plot_processing_nmr.py``.
reference = scp.nmr.read(nmrdir / "topspin_2d" / "1" / "pdata" / "1" / "2rr")

# %%
# Plot the processed TopSpin reference
_ = reference.plot_map()

# %%
# Stage 1: process F2 (direct dimension)
# ---------------------------------------
# ``Experiment.process()`` applies the usual direct-dimension workflow:
# exponential multiplication, FFT, then automatic phasing along F2.
#
# Here we also set the F2 zero-filled size to match the processed TopSpin
# spectrum more closely.

from spectrochempy_nmr import Experiment

exp = Experiment(ser)
f2_processed = exp.process(apodization="em", lb=2.0, size=2048)

# After F2 processing: F2 is in frequency domain, F1 is still time domain
f2_processed

# %%
# Plot after F2 processing — only the horizontal axis is transformed
_ = f2_processed.plot_map()

# %%
# Stage 2: process F1 (indirect dimension)
# -----------------------------------------
# F1 still needs the usual indirect-dimension steps.  We keep them explicit
# here because they are often tuned case by case in practice.
#
# The values below are chosen so that the resulting spectrum is close to the
# processed TopSpin ``2rr`` reference shipped with the test data.
f1 = f2_processed.zf_size(size=1024, dim="y")
f1 = f1.em(lb=5.0, dim="y")
f1 = f1.fft(dim="y")

f1

# %%
# Plot the fully processed 2D spectrum
_ = f1.plot_map()

# %%
# Extract and inspect a 1D slice
# ------------------------------
# Once the 2D spectrum is processed we can reuse the same style of operations
# shown in ``plot_processing_nmr.py``.

slice_f2 = f1[-27.6, :]
_ = slice_f2.plot()

# %%
# Compare the same slice against the TopSpin ``2rr`` reference.
# We normalize both traces before overlaying them because the absolute
# intensity depends on the exact processing chain (window functions,
# phasing, scaling conventions, etc.).
reference_slice_f2 = reference[-27.6, :]
slice_f2_norm = slice_f2.normalize(method="max", dim="x")
reference_slice_f2_norm = reference_slice_f2.normalize(method="max", dim="x")
_ = slice_f2_norm.plot(color="black")
_ = reference_slice_f2_norm.plot(clear=False, color="red", linestyle="--")

# %%
