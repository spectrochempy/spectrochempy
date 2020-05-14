# -*- coding: utf-8 -*-
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

# %% [markdown] {"pycharm": {"name": "#%% md\n", "is_executing": false}}
# # Import IR Data
#
# This tutorial shows the specifics related to infrared data import in Spectrochempy. As prerequisite, the user is expected to have read the [Import Tutorial](Import.ipynb).
#
# Let's first impost spectrichempy

# %% {"pycharm": {"name": "#%%\n", "is_executing": false}, "jupyter": {"outputs_hidden": false}}
import spectrochempy as scp

# %% [markdown]
# # 1. Supported fileformats
#
# At the time of writing this tutorial (Scpy v.0.1.18), spectrochempy has the following readers which are specific to IR data:
#
# - `read_omnic()` to open omnic (spa and spg) files
# - `read_bruker_opus()` to open Opus (*.0, ...) files
# - `read_jdx()` to open an IR JCAMP-DX datafile
#
# and
#
# -`read_csv()'
#     
#     
#  
# [Bruker OPUS](https://www.bruker.com/products/infrared-near-infrared-and-raman-spectroscopy/opus-spectroscopy-software.html) and [Thermo Scientific OMNIC](https://www.thermofisher.com/search/browse/category/us/fr/602580/FTIR%2C+NIR+%26amp%3B+Raman+Software+%26amp%3B+Libraries) softwares have proprietary binary file formats which have been reverse engineered, hence allowing extracting key data. The Omnic reader of Spectrochempy (`read_omnic()`) has been developed based on disussions on the .spa file format in open forums and extended by us to .spg file formats. The Opus reader (`read_bruker_opus()`) is essentially a wrapper of the python module
# [brukeropusreader](https://github.com/qedsoftware/brukeropusreader) developed by QED. 
#
# [JCAMP-DX](http://www.jcamp-dx.org/) is an open format initially developped for IR data and extended to other spectroscopies. At present, the JCAMP-DX reader implemented in Spectrochempy is limited to IR data and AFFN encoding (see R. S. McDonald and Paul A. Wilks, JCAMP-DX: A Standard Form for Exchange of Infrared Spectra in Computer Readable Form, Appl. Spec., 1988, 1, 151–162. doi:10.1366/0003702884428734) fo details.   
#

# %%
