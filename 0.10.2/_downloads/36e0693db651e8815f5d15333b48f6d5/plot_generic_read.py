# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Reading datasets
================

In this example, we show the use of the generic `read` method to create dataset
either from local or remote files.
"""

# %%
# First we need to import the spectrochempy API package
import os
from pathlib import Path

import spectrochempy as scp

TEST_FILE = Path(os.environ.get("TEST_FILE", "irdata/nh4y-activation.spg"))
TEST_FOLDER = Path(os.environ.get("TEST_FOLDER", "irdata"))

# %%
# Import dataset from local files
# -------------------------------
# Read a IR data recorded in Omnic format (``.spg`` extension).
# We just pass the file name as parameter.
dataset = scp.read(TEST_FILE)
print(dataset)

# %%
if dataset is not None:
    _ = dataset.plot(style="paper")

# %%
# When using `read`, we can pass filename as a `str` or a `~pathlib.Path` object.
filename = TEST_FILE
dataset = scp.read(filename)

# %%
# Note that is the file is not found in the current working directory, `SpectroChemPy`
# will try to find it in the ``datadir`` directory defined in `preferences` :
datadir = scp.preferences.datadir
print(datadir)

# %%
# If the supplied argument is a directory, then the whole directory is read at once.
# By default, the different files will be merged along the first dimension (y).
# However, for this to work, the second dimension (x) must be compatible (same size)
# or else a WARNING appears. To avoid the warning and get individual spectra, you can
# set ``merge`` to `False` . This test-data directory may also contain historical
# trusted native `.scp` archives, which require explicit legacy opt-in.
dataset_list = scp.read(TEST_FOLDER, merge=False, allow_unsafe_legacy=True)
print(dataset_list)

# %%
# to get full details on the parameters that can be used, look at the API documentation:
# `spectrochempy.read` .

# %%
# Import dataset from remote files
# --------------------------------
# To download and read file from remote server you can use urls.
try:
    dataset_list = scp.read("http://www.eigenvector.com/data/Corn/corn.mat")
except FileNotFoundError:
    dataset_list = None
    print("Eigenvector corn dataset not reachable; skipping remote import examples.")
else:
    # %%
    # In this case the matlab data contains 7 arrays that have been automatically
    # transformed to `NDDataset` .
    for nd in dataset_list:
        print(f"{nd.name} : {nd.shape}")

    # %%
    # The `eigenvector.com <eigenvector.com>`__ website contains the same data in a
    # compressed (zipped) format:
    # `corn.mat_.zip <https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip>`__ .
    # This can also be used by the `read` method.
    dataset_list = scp.read(
        "https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip"
    )
    print(dataset_list)

    # %%
    # Plot each of the datasets
    _ = dataset_list[-1].plot()
    _ = dataset_list[-2].plot()
    _ = dataset_list[-3].plot()
    _ = dataset_list[-4].plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
