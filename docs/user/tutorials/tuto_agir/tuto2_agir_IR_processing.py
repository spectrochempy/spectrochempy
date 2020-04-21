# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"nbpresent": {"id": "00240045-c477-45dd-94f4-037a0fbd5000"}, "slideshow": {"slide_type": "slide"}}
# # Processing the IR dataset

# %% {"cell_style": "center", "hide_input": false, "nbpresent": {"id": "4187b2b2-7280-42b1-8c0d-a4af39101f17"}, "slideshow": {"slide_type": "subslide"}}
from spectrochempy import *
set_loglevel(ERROR)

# %% [markdown]
# ## Loading a project

# %% [markdown] {"nbpresent": {"id": "af79a3ae-e4ce-4602-8ed8-e590658d0d90"}, "slideshow": {"slide_type": "-"}}
# We read the ``HIZECOKE`` project that was saved previously

# %% {"nbpresent": {"id": "0126e973-e46c-4a4f-bf9e-dccf54347cce"}, "slideshow": {"slide_type": "-"}}
proj = Project.load('HIZECOKE')
proj

# %% [markdown]
# ## Preprocessing the IR spectra

# %% [markdown] {"nbpresent": {"id": "837d22f3-a331-4bc6-a464-6b3cf80a4e26"}}
# Let's replot the data. But we will slice the data in order to keep only the region of interest, between 3990 and 1300 cm$^{-1}$.
#
#
#
# ### Some notes about slicing
#
# Slicing can be done by index, coordinates or labels (when they are present).
#
# * `proj.P350.IR[:, 10]` for column slicing (here we get the 10th column (with index starting at 0!))
# * or `proj.P350.IR[10]` for row slicing
#
# As said above, we can also slice using the real coordinates. For example,
#
# * `proj.P350.IR[:, 3000.0:3100.0]` will select all columns from wavenumbers 3000 to 3100. 
#
# <div class='alert alert-info'>**IMPORTANT**:
#
# * When doing such slicing, the wavenumbers must be expressed as **floating numbers** (with the decimal separator present) or it will fail!.
#
# * Note that when using a range of coordinates, both limits needs to be set as SpectroChemPy cannot infer which direction to take (and so results may be impredictable):  
# This is ok: `proj.P350.IR[:, 3000.0:3100.0]`, but not this: `proj.P350.IR[:, 3000.0:]`
#
# </div>

# %%
for p in proj.projects:   
    p.IR[:, 3990.:1300., INPLACE]         # we slice the data inplace 
    p.IR = (p.IR.T - p.IR[:,3990.]).T     # we remove some offset 

# %% {"nbpresent": {"id": "57b59b4c-a867-44b9-a9a4-98d1756a6a42"}}
datasets =[]
labels = []

for p in reversed(proj.projects):   # we put the parent first this way
    datasets.append(p.IR)
    labels.append(p.label) 
    
_ = multiplot_stack(datasets=datasets, labels=labels, 
                nrow=1, ncol=3, sharex=True, sharez=True, 
                figsize=(6.8,3), dpi=100, style='sans')

# %% [markdown]
# ## Creating and using project's scripts

# %% [markdown] {"nbpresent": {"id": "304c27b4-dbb2-4a59-90a0-1e61285000f8"}}
# In order to avoid having to write this basic pre-processing and plotting function each time we read the initial project, there is two ways:
#
# 1. We can save the mofified data and reopen the project with the modified files next time we need them. However, the processing that was applied cannot be modified.
# 2. The second possibility is to save the processing scripts along with the project, and reapply them next time we will open this project. If a modification is required, any of the scripts can be modified before being applied.
#
# it would be useful to keep the script used here along with the project.
#
# To do so, we will store them in the project, using the `%addscript` magic function.
#
# The syntax is simple!
#
# There are two possibilities: 
#
# 1. `%addscript -p <project_name> -o <scriptname> <cell_reference, file or objects>`
# 1. `%%addscript -p <project_name> -o <scriptname>` 
#     
#         ...code lines...
#
# where
#
# * `-p <project_name>` to say in wich project we want to strore our script
# * `-o <script_name>` to say which name will be used for that script. It must be unique in a project, or confusion may rapidly arise.
#
# That's almost all:
# * The script can come from the current cell (in this case modify the command like this: `%%addscript -p <project_name> -o <script_name>` (yes with two `%`). This should be the first line of the cell to use as a script. 
# * it can also come from another already executed cell: in this case with give the `id` (*e.g.,* `3`) of the cell to use or a range of cells to use (*e.g.,* `3-10`)

# %% {"nbpresent": {"id": "a1ae86af-bf52-49f8-97f5-eae32d3845b2"}}
# %addscript -p proj -o plotIR 4

# %% {"nbpresent": {"id": "a1ae86af-bf52-49f8-97f5-eae32d3845b2"}}
# %addscript -p proj -o preprocess 3

# %% {"nbpresent": {"id": "4bcb0c3f-b26c-4c4e-b918-a2911e247dd6"}}
# %%addscript -p proj -o print_info
print("samples contained in the project are: %s"%proj.projects_names)

# %% [markdown]
# Let's check that the scripts are stored in the project:

# %%
proj

# %% [markdown]
# and that a script can be executed!

# %%
proj.print_info()

# %% [markdown]
# Now, we save the project... but as we want to keep the original data intacts, we will save only the scripts and keep all data unchanged.

# %%
proj.save(overwrite_data=False)

# %% [markdown]
# Let's now reload the project in a new object, and check that this is the original data for the dataset

# %%
#set_loglevel(DEBUG)
proj = Project.load('HIZECOKE')
proj.plotIR()

# %% [markdown]
# We get the original spectra as expected. 
#
# But now we can use the `preprocess` script which is stored in the project, to apply all change that were recorded.

# %% {"run_control": {"marked": false}}
proj.preprocess()
proj.plotIR()

# %% [markdown] {"nbpresent": {"id": "d64f8d55-67db-44a9-a1d3-d95f8f8f0cca"}}
# ## Masking bad data
#
# ### Set a mask on a range of columns
# Clearly some of the spectra above displayed have problem with noise, or some experiment artifacts. 
# Although we could be tempted to simply remove the data, another way, it to mask these bad data. 
#
# For instance, we may want to mask all data ranging between 2300 and 2700. Let's do that:

# %%
for p in proj.projects:   
    p.IR[:, 2300.:2700.]=masked

proj.plotIR()

# %% [markdown] {"nbpresent": {"id": "e3682b70-bede-4108-a2c4-30410480a159"}}
# ### Remove a mask
#
# To remove the mask, we cannot make it selectively. All mask must be removed at the same time:

# %% {"nbpresent": {"id": "fcfbcfdd-efce-4243-89e2-1cbc57f20494"}}
for p in proj.projects:   
    p.IR.remove_masks()

proj.plotIR()

# %% [markdown] {"nbpresent": {"id": "ef14e3b1-4974-4895-bcec-73eeb0ae5664"}}
# ### Masking bad data for rows

# %% [markdown] {"nbpresent": {"id": "4def4553-804c-405f-babc-4063afa5d4d8"}}
# It appears that the data to remove correspond to some row but it is difficult to find which one in the plot above. So it may be interesting to work on transposed data (we use the operator `.T`).
#
# First, let's make a script for plotting the transposed data

# %% {"nbpresent": {"id": "42228607-2aaa-4b18-aa06-c22b87793d97"}}
# %%addscript -p proj -o plotIR_T

datasets_T =[]
labels = []

for p in reversed(proj.projects):   # we put the parent first this way
    datasets_T.append(p.IR.T)
    labels.append(p.label) 
    
_ = multiplot_stack(datasets=datasets_T, labels=labels, 
                nrow=1, ncol=3, sharex=False, sharez=True, 
                figsize=(6.8,3), dpi=100, style='sans')

# %%
proj.plotIR_T()

# %% [markdown]
# Now mask some of the rows, that are clearly not correct

# %% {"nbpresent": {"id": "fa866432-6983-4b1f-bfd1-bf2fa986a77d"}}
proj.P350.IR[14.4:24.]=masked
proj.A350.IR[24.17:24.36]=masked

proj.plotIR_T()
proj.plotIR()

# %% [markdown]
# ## Make our final preprocessing script
#
# OK, the data are now clean. Let's update our `preprocess` script with a summary of all actions, so that we can apply this next time.

# %%
# %%addscript -p proj -o preprocess

# slicing
for p in proj.projects:
    p.IR[:, 3990.:1300., INPLACE]   # we slice the data inplace 
    p.IR = (p.IR.T - p.IR[:,3990.]).T     # we remove the offset 

# mask some rows
proj.P350.IR[14.4:24.]=masked        
proj.A350.IR[24.17:24.36]=masked


# %% [markdown]
# Save these scripts

# %%
proj.save('HIZECOKE', overwrite_data=False)

# %% [markdown]
# In the next [notebook](tuto3_agir_tg_processing.ipynb), we will proceed with some basic pre-processing of the TGA data.
