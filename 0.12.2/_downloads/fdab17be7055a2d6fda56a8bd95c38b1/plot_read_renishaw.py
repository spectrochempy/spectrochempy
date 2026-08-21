# ruff: noqa
"""
Reading Renishaw WiRE files
===========================
This example shows how to read a Renishaw WiRE files (``.wdf`` format).

The spectral data used has been downloaded from
https://github.com/alchem0x2A/py-wdf-reader (MIT License) and stored in the
local directory ``ramandata/wire``.

The examples given here is a modified version of the original examples:
See https://github.com/alchem0x2A/py-wdf-reader/blob/master/examples
"""

# %%
# Import the package
import spectrochempy as scp

# %%
# Read a single spectrum
# ----------------------
dataset = scp.read_wire("ramandata/wire/sp.wdf")
_ = dataset.plot()

# %%
# Read a depth series
# --------------------
dataset = scp.read_wdf("ramandata/wire/depth.wdf")
_ = dataset.plot(method="image")

# %%
# Handle diverging colormaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# A diverging colormap is triggered when negative values are present.
# Enforce a sequential colormap explicitly:
_ = dataset.plot_image(cmap="cividis")

# %%
# Alternatively, adjust ``diverging_margin`` to keep the default colormap:
_ = dataset.plot(method="image", diverging_margin=0.1)

# %%
# Filter blank spectra
# --------------------
import numpy as np

keep_rows = np.where(dataset.data.mean(axis=1) > 0)[0]
dataset = dataset[keep_rows]
_ = dataset.plot_image()

# %%
# Read line scan and grid scan data
# ----------------------------------
# Line scan from a StreamLine HR measurement:
dataset = scp.read("ramandata/wire/line.wdf")
_ = dataset.plot_image()

# %%
# Grid scan data:
dataset = scp.read_wdf("ramandata/wire/mapping.wdf")
_ = dataset.sum(dim=2).plot_image(equal_aspect=True)
_ = dataset[..., 1529.0].plot_image(equal_aspect=True)


# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
