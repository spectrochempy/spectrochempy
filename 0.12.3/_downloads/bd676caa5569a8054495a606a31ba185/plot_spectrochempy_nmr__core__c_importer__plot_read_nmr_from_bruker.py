# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Loading of experimental 1D NMR data
===================================

In this example, we load a 1D Bruker TopSpin NMR dataset and inspect the raw
FID with the public ``scp.nmr.read(...)`` API.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.

"""
# sphinx_gallery_thumbnail_path = 'gettingstarted/examples/gallery/auto_examples_core/c_importer/images/thumb/sphx_glr_plot_read_nmr_from_bruker_thumb.png'

# %%
import spectrochempy as scp

# %%
# `datadir.path` contains the path to a default data directory.

datadir = scp.preferences.datadir

path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d"

# %%
# load the data in a new dataset

ndd = scp.nmr.read(path, expno=1, remove_digital_filter=True)

# %%
# view it...

_ = ndd.plot()

# %%
# The public gallery currently focuses on validated 1D workflows. Raw 2D TopSpin
# examples remain in the repository for later characterization, but they are not
# presented here because multi-dimensional processing is still outside the
# supported public scope.

# %%
# This ends the example! The following line can be uncommented if no plot shows
# when running the .py script with python.

# scp.show()
