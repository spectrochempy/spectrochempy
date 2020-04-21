.. _install_mac:

Installation Guide for Mac and linux
####################################

**Table of contents**

.. contents::
   :local:


Prerequisites
=============

|scpy| requires a working python installation (version 3.6 or higher), C/C++ compiler for some modules, Git to get the
last version of the code.

* `Python <http://www.python.org/>`_
    We highly recommend to install **Miniconda** or **Anaconda** python framework (a much straightforward
    solution!) which is available for most platforms and  the rest of this guide will mainly
    use commands for this distribution.

    Miniconda is much lighter while Anaconda is more complete if you intend using
    python beyond Spectrochempy.

    Go to `Anaconda download page <https://www.anaconda.com/distribution/>`_ or
    `Miniconda download page <https://docs.conda.io/en/latest/miniconda.html>`_.
    Choose your platform and download one of the available installer, *e.g.*, the 3.6 or + version.

    Install the version which you just downloaded, following the instructions on the download page.

    Some modules need a C/C++ compilation. They are in principle present by default in Mac OS and Linux platforms.

Installation
=============

.. _conda_mac:

Create a new conda environment
******************************

For compatibility issues we STRONGLY recommend using a specific conda environment to use |scpy|.
To do so follows the following steps:

#.  Open a terminal and update conda:

    .. sourcecode:: bash

        (base)  ~ $ conda update conda

    you exact prompt may be different depending on the shell you are using and its configuration

#.  Add channels to get specific packages:

    .. sourcecode:: bash

        (base)  ~ $ conda config --add channels conda-forge

#.   Now we can create the `scpy` environment with all the required python packages.

    Download the following configuration file: `scpy.yml <https://bitbucket.org/spectrocat/spectrochempy/downloads/scpy.yml>`_

    .. sourcecode:: bash

        (base)  ~ $ conda env create -f=<DonwloadsFolderPath>>/scpy.yml

#.  Switch to this environment. At this point, `(scpy)` should appear before the prompt instead of `(base)`.

    .. sourcecode:: bash

        (base)  ~ $ conda activate scpy
        (scpy)  ~ $

    Note:

        You can chose to make the `scpy` environment as a default

        Edit the startup profile so that the last line is source activate environment_name.
        In Mac OSX this is ~/.bash_profile, in other environments this may be ~/.bashrc.
        If you use Mac OSX Catalina, it may be ~/.zshrc.

        .. sourcecode:: bash

            $ open ~/.bash_profile

        Go to end of file and type the following:

            source activate scpy

        Save and exit File. Start a new terminal window.
        Type the following to see what environment is active

        .. sourcecode:: bash

            $ conda info -e

        The result shows that your are using your environment by default.

Install of the |scpy| package
*****************************

install the |scpy| package in this environment using one of the following method.

Conda install
-------------

Todo

Install from the Bitbucket source repository
--------------------------------------------

Using this method you can install the latest stable version (`MASTER <https://bitbucket.org/spectrocat/spectrochempy/src/master/>`_)

.. sourcecode:: bash

    (scpy) ~ $ pip install https://bitbucket.org/spectrocat/spectrochempy/get/master.zip

or the latest development version (`DEVELOP <https://bitbucket.org/spectrocat/spectrochempy/src/develop/>`_).

This must be done with caution because in this case instabilities are more likely to occurs than
with the (`MASTER <https://bitbucket.org/spectrocat/spectrochempy/src/master/>`_).
It is recommended to use a different conda environnement in this case

.. sourcecode:: bash

    (scpy) ~ $ pip install https://bitbucket.org/spectrocat/spectrochempy/get/develop.zip

Install a developper version (Advanced usage)
---------------------------------------------

Installation of the developper version is described here:  :ref:`develguide`.


Check the Installation
----------------------

Run a IPython session by issuing in the terminal the following command:

.. sourcecode:: bash

    (scpy) ~ $ ipython

Then execute two commands as following:

.. sourcecode:: ipython

    In [1]: from spectrochempy import *

    In [2]: NDDataset()

If this goes well, the |scpy| application is likely functional.

Jupyter notebook extensions
===========================

After the installation above, to be able to use spectrochempy in notebooks
with the full plotting capabilities we need to execute the  following command:

.. sourcecode:: bash

    (scpy) ~ $ conda install -c conda-forge widgetsnbextension

Jupyter lab extensions
=======================

As for notebooks we need these additional steps (jupytext, jupyterlab-manager and jupyter-matplotlib extensions

.. sourcecode:: bat

    (scpy) ~ $ jupyter nbextension install --py jupytext --user
    (scpy) ~ $ jupyter nbextension enable --py jupytext --user
    (scpy) ~ $ jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

If jupyter lab ask you for building, do it!


Getting started
===============

The recommended next step is to proceed to the |userguide|_


.. _`easy_install`: http://pypi.python.org/pypi/setuptools
.. _`pip`: http://pypi.python.org/pypi/pip
