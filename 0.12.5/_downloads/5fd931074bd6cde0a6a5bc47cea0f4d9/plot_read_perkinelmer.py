# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
# sphinx_gallery_thumbnail_number = 1

"""
Reading a PerkinElmer SP file (plugin)
======================================

This example shows how to read a PerkinElmer ``.sp`` binary IR file using the
optional ``spectrochempy-perkinelmer`` plugin.

.. hint::

   **Requires the official ``spectrochempy-perkinelmer`` plugin.**
   Install with: ``pip install spectrochempy[perkinelmer]``.
"""

# %%
import spectrochempy as scp

# %%
# Read the sample PerkinElmer file
# ---------------------------------
# The PerkinElmer reader is available through the namespaced API
# ``scp.perkinelmer``. A single-spectrum ``.sp`` file is loaded as an
# ``NDDataset`` with wavelength coordinates and available metadata.

dataset = scp.perkinelmer.read("irdata/perkinelmer/spectra.sp")

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

# Uncomment the following line to display all figures when running the script
# directly with Python.
#
# scp.show()
