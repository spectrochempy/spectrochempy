# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Reading a PerkinElmer SP file (plugin)
=========================================

This example shows how to read a PerkinElmer ``.sp`` binary IR file using the
optional ``spectrochempy-perkinelmer`` plugin.

.. hint::

   **Requires the official ``spectrochempy-perkinelmer`` plugin.**
   Install with: ``pip install spectrochempy[perkinelmer]``.
"""
# sphinx_gallery_thumbnail_path = 'gettingstarted/examples/gallery/auto_examples_plot_read_perkinelmer.py/images/thumb/sphx_glr_plot_read_perkinelmer_thumb.png'

# %%
import spectrochempy as scp

# %%
# Read the sample PerkinElmer file
# ---------------------------------
# The PerkinElmer reader is available through the namespaced API
# ``scp.perkinelmer``. A single-spectrum ``.sp`` file is loaded as an
# ``NDDataset`` with wavelength coordinates and available metadata.

datadir = scp.preferences.datadir
filename = datadir / "irdata" / "perkinelmer" / "spectra.sp"

dataset = scp.perkinelmer.read(filename)

# %%
# Display the dataset summary:
dataset

# %%
# The dataset preserves available metadata from the PerkinElmer file:
print(f"Instrument model: {dataset.meta.instrument_model}")
print(f"Detector:        {dataset.meta.detector}")
print(f"Source:          {dataset.meta.source}")
print(f"Analyst:         {dataset.meta.analyst}")
print(f"Date:            {dataset.meta.date}")
print(f"Accumulations:   {dataset.meta.accumulations}")

# %%
# Plot the spectrum:
_ = dataset.plot()

# %%
# This ends the example ! The following line can be removed or commented
# when the example is run as a notebook (`.ipynb`).

# scp.show()
