# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive baseline corrections

# %%
import spectrochempy as scp

# %% [markdown]
# The following line is required to open an external window (the method doesn't work in the notebook)

# %%
# %matplotlib qt

# %%
X = scp.read_omnic("irdata/nh4y-activation.spg")
X[:, 1290.:890.] = scp.MASKED

basc = scp.BaselineCorrection(X)

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
