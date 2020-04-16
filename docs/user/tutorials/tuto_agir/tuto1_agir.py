# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"cell_style": "center", "nbpresent": {"id": "8c1a09c3-41be-4a90-bb14-9ee105fbd2a4"}, "slideshow": {"slide_type": "slide"}}
# # A full working example: an AGIR study of coked zeolites

# %% [markdown] {"cell_style": "center", "nbpresent": {"id": "e4d6f81d-e9ac-48e6-80e3-6ab639743531"}, "slideshow": {"slide_type": "slide"}}
# This example concerns the **study of coked hierarchical zeolites**.
#
# This notebook however focuses on the processing step we have used for this work, as well as some details about the use of the **SpectroChemPy** python package.
#
# Fisrt, and this will be done at the top of each notebook in the tutorial, we load the **SpectroChemPy** API. It will expose all available methods that can act on `projects` and `datasets`.

# %% {"nbpresent": {"id": "7ebe96fc-565c-46f4-bdd4-ca8903347319"}}
from spectrochempy import *
set_loglevel(WARNING)

# %% [markdown] {"nbpresent": {"id": "6045aec1-0300-4f99-91ad-e45b8feffcd7"}, "slideshow": {"slide_type": "subslide"}}
# ## Introduction
#
# The AGIR instrument setup is displayed in the following figure. 
#
# ![figagirsetup](annotated_fig_agir_setup.jpg)
#
# This instrument consists in the combination of an infrared cell and a balance. It allows recording *operando* IR spectra and, simultaneously, the mass of the sample.

# %% [markdown] {"nbpresent": {"id": "0e06a074-383c-4411-82a4-cb4539971d2f"}}
# ## Managing the data in a project and subprojects
#
# The data we will be processing in this tutorial were produced using this AGIR setup at the Laboratory LCS.
#
# We have TGA (Thermogravimetric analysis) and FTIR (Fourier Transform InfraRed spectroscopy) data for 3 samples, which have been recorded simultaneously at 350 °C (623 K) under reaction with methanol:
#
# * P (parent zeolite sample), 
# * A (subjected to an acid treatment), 
# * B (subjected to a basic treatment).
#
# We know that some of the bands that develop on the IR spectra are mainly due to an increase of the coke deposits, which corresponds to an increase of the mass of the pellet. This mass is recorded.
#
# In this work we will try to correlate those evolutions of IR bands and mass of the deposits.

# %% [markdown] {"nbpresent": {"id": "14a6077d-9206-4023-b4f1-d741f50390aa"}}
# To manage all the data for three samples easily, we will create a *project*.
#
# For this we can use the ``Project manager`` included in SpectroChemPy.
#
# First we create a main project containing three sub-projects, one for each sample. A project need at least a project name (*e.g.*, the name of the sample) but if if it is not provided, some automatic title will be generated.
#
# Some metadata can be also added at this stage or later, for example here, we add for each project a `Latex` formatted string that will serve as a label for the plot of the data. In the same way other useful metadata could be added, *e.g.*, a description of the project, etc.
#
# The actual TG and IR data will be added later.

# %% {"nbpresent": {"id": "352318e6-4461-47fc-a0f8-4338ce4f319c"}}
proj = Project(
                Project(name='P350', label=r'$\mathrm{M_P\,(623\,K)}$'),
                Project(name='A350', label=r'$\mathrm{M_A\,(623\,K)}$'),
                Project(name='B350', label=r'$\mathrm{M_B\,(623\,K)}$'),
                name = 'HIZECOKE',
              )

# %% [markdown] {"nbpresent": {"id": "32d224ec-f047-43b8-829a-556e01023899"}}
# Display the structure of this project

# %% {"nbpresent": {"id": "66ad9612-6cc8-4993-a4e8-4cd1be6951a3"}}
proj

# %% [markdown] {"nbpresent": {"id": "addb48d9-2b44-4ffa-a339-f5b7cabbd230"}}
# We can access the content and attributes of a Project object using 2 syntax. One is the following (which works only if the name of the object doesn't contain any space, dash, dot, or other non alphanumeric characters, excepted the underscore "`_`". Also, do not start a name with a number.):

# %% {"nbpresent": {"id": "2620f592-af9c-40f3-a4da-3eeb617cf346"}}
label = proj.P350.label
label

# %% [markdown] {"nbpresent": {"id": "f7c40a67-f94b-492c-9103-ef85a0a0a4e1"}}
# As the label is a latex string, we can use the `Latex` command get a prettier output of the latex content for this label

# %% {"nbpresent": {"id": "6e8d57c6-9e05-4e1c-b090-8af20fe33b84"}}
Latex(label)  # note that this work only on the real jupyter notebook (not in the HTML documentation)

# %% [markdown] {"nbpresent": {"id": "a82ec5f8-f33a-4077-961c-f117a3565fa8"}}
# If the above syntax doesn't work (e.g., it the name contains space), we can always access using keys (Project are actually behaving mainly as python dictionaries)

# %% {"nbpresent": {"id": "0447056e-7e6b-49c0-83f2-34e3c468c69a"}}
Latex(proj['P350'].label)

# %% [markdown] {"nbpresent": {"id": "b6db65a9-efc2-4489-b158-a7adb30e83c8"}}
# ## Reading the experimental data

# %% [markdown] {"nbpresent": {"id": "617d191e-c90b-4457-a2f4-30d942690b2c"}}
# The IR data have been exported as `csv` file (on file per spectra) and then zipped. Therefore, we will use the function `read_zip` to open the IR data. 
#
# For the TGA data are in `csv` files, so we use `read_csv` to read them.

# %% [markdown] {"nbpresent": {"id": "80f7c86f-60ac-40cc-9ce3-262c7fdde02d"}}
# ### Reading the raw  IR data
#
# Note that reading a large set of `.csv` file is rather slow. Be patient! 
#
# After reading each dataset, we save the data in the `scp` format of SpectroChemPy (extension `.scp`, so that any further reading of the data will be much faster.
#
# For this tutorial, the experimental data re stored in a example directory, accessed here through, the variable `general_preferences.datadir`. As we have to repeating the same kind of reading three times, let's write a small reading function:

# %% {"nbpresent": {"id": "10466eb8-8fa0-483d-9f88-b78e1368d1b3"}}
# Our data are in our example directory. 
datadir = general_preferences.datadir
    
def readIR(name, force_read_original=False):
    
    # Here is the full path (depending on name) without the file extension:
    basename = os.path.join(datadir,'agirdata','{}'.format(name),'FTIR','FTIR')
    
    # Check first if a `.scp` file have already been saved.
    # This happens if this not the first time we are executing this notebook.
    # If it exists we will use this file which is much faster to read.
    if os.path.exists(basename+'.scp') and not force_read_original:
        # A `.scp` file allready exists, use this one (except if the flag
        # `force_read_original` is set):
        filename = basename + '.scp'
        # we use the generic `read` method of the class NDDataset 
        # to read this file and create a new dataset.
        dataset = NDDataset.read( filename)
        
    else:
        # This is the first time we execute this function, 
        # then we read the original `.zip` file
        filename = basename + '.zip'
        # to read it, we use the `read_zip` method:
        dataset = NDDataset.read_zip( filename, origin='omnic_export')
        # to accelerate the process next time, we save this dataset 
        # in a `scp` format
        dataset.save(basename + '.scp')
        
    return dataset


# %% [markdown] {"nbpresent": {"id": "9aaf6138-8601-4b8d-bb9a-0a1bd021c475"}}
# Ok, now we use this function to read the 3 FTIR file and we add them to the corresponding sub-project of our project

# %% [markdown] {"nbpresent": {"id": "a622712c-d5df-405b-b95f-353fd8b9245f"}}
# The list of names of the subprojects is accessible in the property `projects_names` of the dataset.

# %% {"nbpresent": {"id": "00e09b98-a48a-4ffa-9f9a-f4ecee874d37"}}
for name in proj.projects_names:
    # read the corresponding dataset
    dataset = readIR(name, force_read_original=False)
    # add it to the corresponding sub-project. We can access directy using its name
    # NOTE: all names of objects contained in a given project 
    # must be unique for this to work!
    # Last thing. because the name of the Dataset is not very informative 
    # we give explicitely a name for the entry in the subproject
    proj[name].add_dataset(dataset, name='IR')

# %% [markdown] {"nbpresent": {"id": "31de88d7-4261-445a-9918-df6775eb9907"}}
# Display the updated structure of the project:

# %% {"nbpresent": {"id": "bb61f400-a242-47a4-b0ff-78f2aa14b451"}}
proj

# %% [markdown]
# Let's look at one of the dataset

# %%
proj.P350.IR

# %% [markdown] {"nbpresent": {"id": "3ec263c9-e9b9-4d2b-a78a-d4cc50a86569"}}
# ### Plot of the raw IR data 

# %% [markdown]
# We can plot these IR data using the `plot_stack` method. Note: this method (as other plot methods) return some values. So to avoid their display in the output we affect them to some variables, e.g., "`_`" if we do not want to keep the results

# %%
p = proj.P350.IR.plot_stack()

# %% [markdown]
# In such 2D stacked plot, the `x` axis corresponds to coordinate `1` and the `y` axis to coordinate `0`.
# This is because the array of data is arranged with dimensions as follow `[y, x]`. The adsorbance (or `z` axis) corresponds to the valueof each element in the array. Sometimes it can be easier to look at the data as a map or an image.

# %%
p = proj.P350.IR.plot_map()

# %%
p = proj.P350.IR.plot_image()

# %% [markdown]
# ### Multiplot of the raw IR data 

# %% [markdown]
# In our project, the data have the same structure: three 2D IR spectra. We can plot them using the `multiplot_stack` function.

# %% [markdown] {"nbpresent": {"id": "6de1226b-7b75-4e03-b1ce-82c9b7df624e"}}
# We need to select the datasets we want to plot and the correspondings labels

# %% {"nbpresent": {"id": "d02d88d0-cac8-486c-adc6-daab89652fc1"}}
datasets = []
labels = []
for p in proj.projects:
    datasets.append(p.IR)
    labels.append(p.label)   

# %% [markdown]
# Now we have two list containing the datasets and their corresponding labels

# %%
datasets

# %%
labels

# %% [markdown] {"nbpresent": {"id": "4f12fc98-2120-4a95-8101-66a3687a146d"}}
# Now we call the `multiplot_stack` function to plots the datasets in row. Note the use of `sharez` , because the vertical axis here is `z` (the intensity) and not `y` (which refer to the evolution) (see above). 

# %% {"nbpresent": {"id": "f8a96a51-2ca0-4c1f-ba85-3dd46899c36c"}}
_ = multiplot_stack(datasets=datasets, labels=labels, 
                nrow=1, ncol=3, sharex=False, sharez=True, 
                figsize=(6.8,3), dpi=100)

# %% [markdown] {"nbpresent": {"id": "4e26d490-9925-4077-9aa3-0be86d846f98"}}
# May be to analyse more easily the `y` dimension, map plot would be better. 
#
# Let's then plot the data using this plot method (plot method: `map` instead of `stack`)
#
# Note:
#     This work only if latex is installed

# %% {"nbpresent": {"id": "08de904a-eacb-4bcf-afde-844c68b78f87"}}
_ = multiplot_map(datasets=datasets, labels=labels, 
                nrow=1, ncol=3, sharex=False, sharey=True, 
                figsize=(6.8,3), dpi=100, style='serif') 

# %% [markdown] {"nbpresent": {"id": "5ba758c4-5eb5-4eaa-99c1-79899bb42241"}}
# Hum? this is not very nice, because the `y` scale being the timestamps (the time from some reference date of the acquisition of the data), they cannot be easily compared.
#
# Let's take a common origin (and express the units in hours instead of seconds)!

# %% {"nbpresent": {"id": "e6e5307c-f948-49e6-a80c-6dfcc23fb34d"}}
for ds in datasets:
    ds.y -= ds.y[0]    # remove the timestamp offset from the axe y
    ds.y.ito('hour')   # adapt units to the length of the acquisition 
                       # for a prettier display
    ds.y.title = 'Time-on-stream' 
                       # put some explicit title (to replace timestamps)

# %% [markdown] {"nbpresent": {"id": "0d262e59-9a75-4bc2-85c6-ae234325424e"}}
# Now we replot the data:

# %% {"nbpresent": {"id": "b7ba84e3-436d-4604-a055-66b9661613c7"}}
_ = multiplot_map(datasets=datasets, labels=labels, 
                nrow=1, ncol=3, sharex=False, sharey=True, 
                figsize=(6.8,3), dpi=100, style='sans') #<--'Sans-serif style (default)'


# %% [markdown] {"nbpresent": {"id": "f2f2d6a1-4e39-43eb-ac49-2be9c17aa5a0"}}
# This is much better!

# %% [markdown] {"nbpresent": {"id": "b27eadf5-931a-4cfb-a646-0360f3f4feb2"}}
# ### Reading the raw TGA data
#
# Now read the TGA data. And again we will create a reading function to apply to the 3 samples.

# %% {"nbpresent": {"id": "afe7d1ed-566d-473b-9db9-3e0464c4ecfd"}}
def readTGA(name, force_read_original=False):
    basename = os.path.join(datadir,'agirdata','{}'.format(name),'TGA','tg')
    if os.path.exists(basename+'.scp') and not force_read_original:
        # check if the scp file have already been saved
        filename = basename + '.scp'
        dataset = NDDataset.read( filename)
    else:
        # else read the original csv file
        filename = basename + '.csv'
        dataset = NDDataset.read_csv(filename)
        dataset.save(basename + '.scp')
    return dataset


# %% [markdown] {"nbpresent": {"id": "fbeb48f0-6db5-4f38-b974-3e1e88141cfc"}}
# Now we read the TGA datasets and add them to our current project

# %% {"nbpresent": {"id": "4b8c99b8-543d-414f-86b9-a4a20358c1ec"}}
for name in proj.projects_names:
    # read the corresponding dataset
    dataset = readTGA(name, force_read_original=False)
    # add it to the corresponding sub-project. 
    proj[name].add_dataset(dataset, name='TGA')

# %% [markdown] {"nbpresent": {"id": "8442c958-ff1c-4a93-8c84-fd20943fd3b0"}}
# Display the updated structure:

# %% {"nbpresent": {"id": "daf71e59-62fc-4269-8c4f-4b2b28ce4c40"}}
proj

# %% [markdown] {"nbpresent": {"id": "2866ee37-9b7b-4ddd-a8f7-3fd9fd574866"}}
# ### Plot of the raw TGA data

# %% [markdown] {"nbpresent": {"id": "f3b2c715-6f44-46ca-8f88-7aa77b146095"}}
# Let's plot the 3 1D TGA datasets on the same figure.

# %% {"nbpresent": {"id": "3be77f88-bd4c-447d-950a-ccf1da4f72ed"}}
tga_datasets =[]
for p in proj.projects:
    tga_datasets.append(p.TGA) 

# %% {"nbpresent": {"id": "999b54fd-d3cf-4ffd-b251-4d28da340e3b"}}
_ = plot_multiple(datasets=tga_datasets, labels=labels, pen=True, style='sans', 
                  markevery=50, markersize=7,
                  legend='lower right')

# %% [markdown] {"nbpresent": {"id": "7a2d363c-ad8f-41d1-a8da-ec903d0bf931"}}
# Some information on this data are not present in the csv files used, so we need to add them now, to get better plots (also some part of the data, *e.g.*, before time 0, are not very useful and will be removed)

# %% {"nbpresent": {"id": "4fbd6096-bf08-45b3-9eaf-e2be419bee99"}}
for tgads in tga_datasets:       
    tgads = tgads[-0.5:35.0]          # keep only useful range. 
                                       # Note the slicing by location
    tgads.x.units = 'hour'             # x dimension units
    tgads.units = 'weight_percent'     # data units
    tgads.x.title = 'Time-on-stream'   # x title information for labeling
    tgads.title = 'Mass change'        # data title information for labeling

# %% [markdown] {"nbpresent": {"id": "bc5863e4-def2-4485-84bd-c5eb439f419b"}}
# We get some warnings because the axis range is too high for two of the dataset. We can safely ignore this.
#
# Note that we can easily remove such warnings by changing the loglevel temporarilly:

# %% {"nbpresent": {"id": "5d61baf9-e4e4-4477-9c34-2329e273648f"}}
set_loglevel(ERROR)

for tgads in tga_datasets:       
    tgads = tgads[-0.5:35.0]          
    tgads.x.units = 'hour'            
    tgads.units = 'weight_percent'    
    tgads.x.title = 'Time-on-stream'  
    tgads.title = 'Mass change'        
    
set_loglevel(WARNING)

# %% [markdown] {"nbpresent": {"id": "d0e300bb-5436-4357-a447-0198ee8cce18"}}
# Let's now plot the data again:

# %% {"nbpresent": {"id": "8802705f-6dc5-4a06-b163-25b2bc1a6e67"}}
_ = plot_multiple(datasets=tga_datasets, labels=labels, pen=True, style='sans', markevery=50, markersize=7,
                  legend='lower right')

# %% [markdown] {"nbpresent": {"id": "f2018a33-66c5-49d3-8ff2-da427f33d3fa"}}
# **Why nothing has changed?**
#
# Simply because when we have done the slicing we have actually created a **copy** of the initial object 
# before doing the slicing. So the original object has not been changed, but the copy!
#
# **How to solve this problem ?**
#
# We need to do the slicing in place and for this just put the constant **INPLACE** in the slicing indications.

# %% {"nbpresent": {"id": "45812edb-5347-4507-bdbb-b70807ced787"}}
set_loglevel(ERROR)
    
for tgads in tga_datasets:       
    tgads[-0.5:35.0, INPLACE]          # keep only useful range. Note the slicing by location
                                       # NOTE: the INPLACE addition!
    tgads.x.units = 'hour'             # x dimension units
    tgads.units = 'weight_percent'     # data units
    tgads.x.title = 'Time-on-stream'   # x title information for labeling
    tgads.title = 'Mass change'        # data title information for labeling
    
set_loglevel(WARNING)

_ = plot_multiple(datasets=tga_datasets, labels=labels, pen=True, style='sans', markevery=50, markersize=7,
                  legend='lower right')

# %% [markdown] {"nbpresent": {"id": "337a475f-b929-4cc6-8740-61273f51bf94"}}
# ### Saving the project

# %% [markdown] {"nbpresent": {"id": "a483ef45-cca7-443e-a1df-3f29d1e96bfc"}}
# Ok, we have now build our project with some data and attributes. 
#
# If we go to another notebook, we would like to get all the data without doing again the operations made on this notebook. 
#
# So we need to save this project.

# %% {"nbpresent": {"id": "81fecd85-1420-4e3f-8597-5545c570797f"}}
proj.save('HIZECOKE')

# %% [markdown]
# In the next [notebook](tuto2_agir_IR_processing.ipynb), we will now proceed with some basic pre-processing of the IR data, such as slicing interesting regions, and masking some data. 
