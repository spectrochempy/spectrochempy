# ----------------------------------------------------------------------------------------------------------------------
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ----------------------------------------------------------------------------------------------------------------------

# %% [markdown]
# # Processing of the TGA data

# %%
from spectrochempy import *
set_loglevel(ERROR)

# %% [markdown]
# First we read our project 

# %%
proj = Project.load('HIZECOKE')
proj

# %%
datasets =[]
labels = []
for p in proj.projects:
    datasets.append(p.TGA) 
    labels.append(p.label)
    
_ = plot_multiple(datasets=datasets, labels=labels, pen=True, style='sans', 
                  markevery=1, markersize=3,
                  legend='lower right')

# %% [markdown]
# ## Initial corrections of the TG data. 
#
# We are interested by the gain in weight during the reaction. But, the zero in time is obviously not the same for all three samples. 
#
# We will now correct this.
#
# We fist display an expansion around the zero region

# %%
_ = plot_multiple(datasets=datasets, labels=labels, pen=True, style='sans', 
                  markevery=1, markersize=3,
                  legend='lower right', xlim=(-0.05,0.2), ylim=(-.1, 2.5))

# %% [markdown]
# Now we correct the time origin for each sample separately, and mask data below 0 in time

# %%
proj.P350.TGA.x -= 0.0577 *ur.hour  # note the use of units as the x data have units!
proj.B350.TGA.x += 0.0093 *ur.hour
proj.A350.TGA.x -= 0.1253 *ur.hour

# %%
for p in proj.projects:
    p.TGA[-0.005:35.0, INPLACE]   # slicing to keep only data for x>0 
    # also set the first point to 0,0!
    p.TGA.x[0] = 0 * ur.hour
    p.TGA[0] = 0 * p.TGA.units
    
# finally we mask some data that seems not correct
#proj.B350.TGA[0.0147] = masked

# %%
_ = plot_multiple(datasets=datasets, labels=labels, pen=True, style='sans', 
                  markevery=50, markersize=7,
                  legend='lower right')

# %% [markdown]
# As in the preprocessing of IR data, let's store two script for displaying the data and make the intial preprocessing:

# %%
# %%addscript -p proj -o preprocessTG 

proj.P350.TGA.x -= 0.0577 *ur.hour  # note the use of units as the x data have units!
proj.B350.TGA.x += 0.0093 *ur.hour
proj.A350.TGA.x -= 0.1253 *ur.hour
for p in proj.projects:
    p.TGA[-0.005:35.0, INPLACE]   # slicing to keep only data for x>0 
    # also set the first point to 0,0!
    p.TGA.x[0] = 0 * ur.hour
    p.TGA[0] = 0 * p.TGA.units

# %%
proj.save('HIZECOKE', overwrite_data=False)
