.. _install:

Installation Guide
###################

**Table of contents**

.. contents::
   :local:

Requirements
============

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

* C/C++ Tools

  Some modules need a C/C++ compilation. They are present by default in Mac OS and Linux platforms. For
  windows, download and install `Build Tools for Visual Studio <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_

* git

  Check whether git is installed.

    * WIN: open a command prompt (Select the Start button and type cmd), or preferably open the Anaconda Prompt or
      Powershell Prompt in the Anaconda start Menu) and type the command:

    .. sourcecode:: bat

        C:\<yourDefaultPath>> git --version

    if this returns an error, install the `command line version <https://git-scm.com/download/win>`_ of git.

    * MAC/OS, LINUX: by default Git is installed on Linux and macOS computers as a command line option.

Installation
=============

.. _conda:

Installation using Conda and git
*********************************

WINDOWS
-------

#.  Open a command prompt (Select the Start button and type cmd), or preferably open the Anaconda Prompt or
    Powershell Prompt in the Anaconda start Menu.

#.  Update conda (yes, even if you have just installed the distribution...):

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda update conda

    where `<yourDefaultPath>` is you default workspace directory (e.g. `C:\\Users\\<user>`)

#.  Add channels to get specific packages:

    .. sourcecode:: bat

        C:\<yourDefaultPath>> conda config --add channels conda-forge

#.  If necessary create your installation directory or go to it.

    We recommend NOT to name it `spectrochempy` because two nested folders `spectrochempy` will also be created at
    the install. You would have then 3 nested `spectrochempy` folders...

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> mkdir <yourInstallDirectory>
        (base) C:\<yourDefaultPath>> cd <yourInstallDirectory>

#.  Clone spectrochempy in your installation directory:

        (base) C:\<yourInstallDirectory>> git clone --depth 1 https://bitbucket.org/spectrocat/spectrochempy.git

    This may take few minutes, go and get your favorite drink or whatever else pleases you...

#.  For compatibility issues we STRONGLY recommend using a specific conda environment to use |scpy|.
    To do so go in the `spectrochempy` directory and create the scpy environment:

    .. sourcecode:: bat

        (base) C:\<yourInstallDirectory>\spectrochempy> cd spectrochempy
        (base) C:\<yourInstallDirectory>\spectrochempy> conda env create -f env/scpy.yml

    This also takes time. Go and get second favorite drink, etc... while the package download and
    extraction proceeds...

#.  Switch to this environment:

    .. sourcecode:: bat

        (base) C:\<yourInstallDirectory>\spectrochempy> conda activate scpy

#.  At this point, `(scpy)` should appear before the prompt. Then install the spectrochempy package in this environment:

    .. sourcecode:: bat

        (scpy) C:/<your installdir>/spectrochempy> pip install .


MAC OS, LINUX
-------------
#. Open a terminal and update conda:

.. sourcecode:: bash

   $ conda update -n base conda

#.  Add channels to get specific packages:

.. sourcecode:: bash

   $ conda config --add channels conda-forge

#.  If necessary create your installation directory or go to it.

    We recommend NOT to name it `spectrochempy` because two nested folders `spectrochempy` will also be created at
    the install. You would have then 3 nested `spectrochempy` folders...

    .. sourcecode:: bash

        $ mkdir <yourInstallDirectory>
        $ cd <yourInstallDirectory>

#.  For compatibility issues we STRONGLY recommend using a specific conda environment to use |scpy|.
    To do so go in the `spectrochempy` directory and create the scpy environment:

.. sourcecode:: bash

   $ conda env create -f=env/scpy.yml

#.  Switch to this environment:

    .. sourcecode:: bash

        $ conda activate scpy

#.  At this point, `(scpy)` should appear before the prompt. Then install the spectrochempy package in this environment:

    .. sourcecode:: bash

        (scpy) $ pip install .



Check the Installation
======================

Run a IPython session by issuing in the terminal the following command:

.. sourcecode:: bash

    $ ipython

Then execute two commands as following:

.. sourcecode:: ipython

    In [1]: from spectrochempy import *

    In [2]: NDDataset()

If this goes well, the |scpy| application is likely functional.

Jupyter notebook
================

After the installation above, to be able to use spectrochempy in notebooks
with the full plotting capabilities we need to execute the  following command:

.. sourcecode:: bash

    $ conda install -c conda-forge widgetsnbextension

Jupyter lab
===========

As for notebooks we need these additional steps:

.. sourcecode:: bash

    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
    $ jupyter labextension install jupyter-matplotlib

If jupyter lab ask you for building, do it!


Getting started
===============

The recommended next step is to proceed to the |userguide|_


.. _`easy_install`: http://pypi.python.org/pypi/setuptools
.. _`pip`: http://pypi.python.org/pypi/pip




