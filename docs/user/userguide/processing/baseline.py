# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline corrections

# %%
import spectrochempy as scp
import os

# %%
dataset = scp.NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
dataset

# %%
# # %matplotlib qt
s = dataset[:, 1260.0:5999.0]
s = s - s[-1]

# Important note that we use floating point number
# integer would mean points, not wavenumbers!

basc = scp.BaselineCorrection(s)

ranges = [[1261.86, 1285.89],
          [1556.30, 1568.26],
          [1795.00, 1956.75],
          [3766.03, 3915.81],
          [4574.26, 4616.04],
          [4980.10, 4998.01],
          [5437.52, 5994.70]]  # predefined ranges

_ = basc.run(*ranges, method='multivariate',
             interpolation='pchip',
             npc=5,
             figsize=(6, 6),
             zoompreview=4)

# %% [markdown]
# The regions used to set the baseline are accessible using the `ranges` attibute:

# %%
ranges = basc.ranges
print(ranges)

# %%
_ = basc.corrected.plot_stack()

# %%
