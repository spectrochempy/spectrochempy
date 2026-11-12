# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Introduction to the plotting library
====================================

This short gallery example shows four common ideas:

- the default ``dataset.plot()`` entry point;
- per-call style changes that do not mutate later plots;
- ``plot_multiple()`` overlaying several 1D datasets on one shared axes;
- saving any of them to a file with ``output``.
"""

import spectrochempy as scp

# %%
# Locate and load a dataset
# --------------------------
dataset = scp.read("irdata/nh4y-activation.spg")

# %%
# Default plot and style
# -----------------------
ax = dataset[0].plot()

# %%
# Per-call style changes
# -----------------------
# Apply a style to this single plot only:
ax = dataset[0].plot(style="classic")

# %%
# The style change is local — the default style is used again here:
ax = dataset[0].plot()

# %%
# Overlay with ``plot_multiple``
# -------------------------------
dataset = dataset[:, ::100]

sample_indices = [index * (dataset.shape[0] - 1) // 4 for index in range(5)]
datasets = [dataset[index] for index in sample_indices]
labels = [f"sample {index}" for index in sample_indices]

_ = scp.plot_multiple(method="scatter", datasets=datasets, labels=labels, legend="best")

# %%
# Saving a figure with ``output``
# -------------------------------
# The ``output`` argument writes the finished figure to a file: the whole
# figure, legend included, written before any display step. ``str`` and
# ``pathlib.Path`` are both accepted, and the format follows the extension.

from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as tmpdir:
    _ = scp.plot_multiple(
        method="scatter",
        datasets=datasets,
        labels=labels,
        legend="best",
        output=Path(tmpdir) / "overlay.png",
        show=False,
    )
    # `output` also works on `dataset.plot` and on `multiplot`.

# %%
# The style change applies only to this call:
_ = scp.plot_multiple(
    method="scatter", style="sans", datasets=datasets, labels=labels, legend="best"
)

# %%
# The default style is used again on the next call:
_ = scp.plot_multiple(method="scatter", datasets=datasets, labels=labels, legend="best")

# %%
# Plotting options
# ----------------
# - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_core_d_plotting_plot_1d_options.py`
# - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_core_d_plotting_plot_2d_projections.py`

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
