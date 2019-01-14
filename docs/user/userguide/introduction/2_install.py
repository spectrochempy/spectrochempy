# ---
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
# ---

# %% [markdown]
# # Installation Guide

# %% [markdown]
# ## Requirements

# %% [markdown]
# |scpy| requires a working python installation.
#
# * `Python <http://www.python.org/>`_
#
# .. warning::
#
# 	Only the python version 3.6 and 3.7 have been tested so far.
#
#
# The following libraries are also required such as :
#
# * `Numpy <http://numpy.scipy.org>`_
#
# * `Scipy <http://www.scipy.org/>`_
#
# * `Matplotlib <http://matplotlib.sourceforge.net/index.html>`_
#
#
# Follow the instructions to install these packages on those sites, or, far easier,
# install them as packages from your operating system
# (e.g. apt-get or the synaptic GUI on Ubuntu, `Macports <http://www.macports.org/>`_ on OS X, etc.).
#
# Regarding the installation of all these above packages, we highy recommend to install **anaconda** or **canopy** python framework (a much straitforward solution!)
# which is available for most platforms.
#
# We prefer the **anaconda distribution**, so tha the rest of this guide will mainly
# use commands for this distribution.
#
# * install the **Anaconda Scientific Python Distribution** : Go to `http://continuum.io/downloads <http://continuum.io/downloads>`_ and follow the instructions for your platform
#
#   If you register as academic member of the university you get interesting
#   add-ons.
#
#   Finally, see :ref:`conda` for installing the |scpy| conda package.
#
# If you prefer **Canopy** here it is:
#
# * install the **Canopy Enthought Distribution** :Go to `https://store.enthought.com/downloads/ <https://store.enthought.com/downloads/>`_ (it's a commercial distribution, but you have a free version Canopy Express.
#
#   If you register with your academic email, you can get the full academic version).
#
#   Finally, see :ref:`pypi` for installing |scpy| .

# %% [markdown]
# ## Installation

# %% [markdown]
# ### Where to Get SpectroChemPy?

# %% [markdown]
# To get a working installation of |scpy| , on any platform (windows, mac OS X, Linux ... ),
# several solutions are (*or will be soon*) available.
#
# * :ref:`pypi`
#
# * :ref:`conda`
#
# * :ref:`dev`
#
# * :ref:`clone`

# %% [markdown]
# #### Standard installation using Conda

# %% [markdown]
# Very simple, use the following command in a terminal:
#
# .. sourcecode:: bash
#
#    $ conda config –add channels conda-forge
#    $ conda config –add channels fernandezc  #TODO: create a channel spectrocat
#
# This will add the necessary channels where to find the various libraries used by |scpy| .
#
# Next we it is recommended to use a dedicated environment for |scpy|, called
# for example `scp37` (using python 3.7)
#
# .. sourcecode:: bash
#
#    $ conda create -n scp37 python=3.7
#    $ source activate scp37
#
# Next one can install |scpy|:
#
# .. sourcecode:: bash
#
#    $ conda install spectrochempy
#
# .. note::
#
#    the conda installer doesn't exist right now. It will be present for version
#    1.b1.
#
#    For now, use the method described for developper, here: :ref:`develguide`

# %% [markdown]
# #### Standard Installation from PyPi using pip

# %% [markdown]
# Again very simple, use the following command in a terminal:
#
# .. sourcecode:: bash
#
#    $ pip install spectrochempy
#
# or to update a previous installation with the latest stable release:
#
# .. sourcecode:: bash
#
#    $ pip install -U spectrochempy
#
# .. note::
#
#    the Pypi installer doesn't exist right now. It will be present for version
#    1.b1.

# %% [markdown]
# #### Installation from zip archives

# %% [markdown]
# Downloads of zip source archives working for all platforms are available, on Bitbucket
#
# .. include:: download.rst
#
# or on PyPi:
#
# * `Download tar.gz archives from PyPi <http://pypi.python.org/pypi/spectrochempy>`_
#
# Ungzip and untar the source package and execute:
#
# .. sourcecode:: bash
#
# 	$ cd <directory>    # replace <directory> by the actual download folder
# 	$ pip install .
#
# or better :
#
# .. sourcecode:: bash
#
# 	$ pip install -e .
#
# to install it in the developper mode.
#
# .. tip::
#
# 	On most UNIX-like systems, you’ll probably need to run these commands as
# 	root or using sudo.
#
# .. _clone:

# %% [markdown]
# #### Clone or Fork of the Bitbucket Repository

# %% [markdown]
# Alternatively, you can make a clone/fork of the github sources at:
#
# * `https://bitbucket.org/spectrocat/spectrochempy  <https://bitbucket.org/spectrocat/spectrochempy>`_
#
# This is the recommended solution for developpers
# and those who would like to contribute (see :ref:`develguide`)

# %% [markdown]
# ### Check the Installation

# %% [markdown]
# Run a IPython session by issuing in the terminal the following command:
#
# .. sourcecode:: bash
#
# 	$ ipython
#
# Then execute two commands as following:
#
# .. sourcecode:: ipython
#
# 	In [1]: from spectrochempy import *
#
# 	In [2]: NDDataset()
#
# If this goes well, the |scpy| application is likely functional.

# %% [markdown]
# ##Getting started
# ===============

# %% [markdown]
# The recommended next step is to proceed to the :ref:`userguide`
#
#
# .. _`easy_install`: http://pypi.python.org/pypi/setuptools
# .. _`pip`: http://pypi.python.org/pypi/pip

# %% [markdown]
# ## Additional notes

# %% [markdown]
# ### Install ipyparallel in Jupyter notebook

# %% [markdown]
# Before using parallel computing in the notebbok, one have to install the module
# from pypi or conda
#
# .. sourcecode:: bash
#
# 	$ conda install ipyparallel
#
# To enable the IPython Clusters tab in Jupyter Notebook:
#
# .. sourcecode:: bash
#
# 	$ ipcluster nbextension enable
# 	$ ipcluster start
