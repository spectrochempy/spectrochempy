# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Introduction to the plotting library
====================================

This short gallery example shows three common ideas:

- the default ``dataset.plot()`` entry point;
- per-call style changes that do not mutate later plots;
- ``plot_multiple()`` overlaying several 1D datasets on one shared axes.
"""

# %%

from os import environ

import numpy as np
import spectrochempy as scp

# %%
# The location of the spectrochempy_data can be found in preferences.

datadir = scp.preferences.datadir


def _load_demo_dataset():
    test_file = environ.get("TEST_FILE")
    if test_file:
        dataset = scp.read(test_file)
        if dataset is not None:
            return dataset

    dataset = scp.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
    if dataset is not None:
        return dataset

    x = scp.Coord(
        np.linspace(4000.0, 650.0, 256),
        title="wavenumber",
        units="cm^-1",
    )
    y = scp.Coord(np.linspace(0.0, 5.0, 16), title="time on stream", units="hour")
    xv = np.linspace(-1.0, 1.0, 256)
    yv = np.linspace(0.0, 1.0, 16)[:, None]
    data = np.exp(-(((xv + 0.35) / 0.12) ** 2)) * (1.0 + 0.5 * yv) + 0.7 * np.exp(
        -(((xv - 0.10) / 0.18) ** 2)
    ) * (1.2 - 0.4 * yv)
    return scp.NDDataset(data, coordset=[y, x], units="a.u.", title="absorbance")


# %%
# Let's read one dataset (in ``.spg`` OMNIC format).

dataset = _load_demo_dataset()

# %%
# First use the default plotting entry point and default style.

ax = dataset[0].plot()

# %%
# Apply a different style to this single plot only.

ax = dataset[0].plot(style="classic")

# %%
# Style selection is local to the previous call, so the default style is used
# again here.
ax = dataset[0].plot()

# %%
# ``plot_multiple()`` overlays several 1D datasets on one shared axes.
dataset = dataset[:, ::100]

sample_indices = np.linspace(0, dataset.shape[0] - 1, 5, dtype=int)
datasets = [dataset[index] for index in sample_indices]
labels = [f"sample {index}" for index in sample_indices]

# Use ``method="scatter"`` when the visual intent is marker-based.
_ = scp.plot_multiple(method="scatter", datasets=datasets, labels=labels, legend="best")

# %%
# As above, the style change applies only to this call.
_ = scp.plot_multiple(
    method="scatter", style="sans", datasets=datasets, labels=labels, legend="best"
)

# %%
# The default style is used again on the next call.
_ = scp.plot_multiple(method="scatter", datasets=datasets, labels=labels, legend="best")

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
