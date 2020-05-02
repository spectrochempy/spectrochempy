.. _install_win:

Installation Guide for Windows
##############################

**Table of contents**

.. contents::
   :local:


Prerequisites
=============

|scpy| requires a working python installation (version 3.6 or higher).

* `Python <http://www.python.org/>`_
    We highly recommend to install  **Anaconda** or **Miniconda** distributions which are available for most
    platforms and  the rest of this guide will mainly use commands for this distribution.

    Miniconda is lighter (400 MB disk space) while Anaconda (3 GB minimum disk space to download and install)
    is much more complete for scientific applications if you intend using python beyond Spectrochempy. Important
    package in Anaconda are also required for spectrochempy (e.g. Matplotib, Numpy, Scipy, Jupyter, …) They are not
    included in Miniconda and will be installed anyway. So overall, the difference in installation time/disc space
    won’t be that big whether you choose Miniconda or Anaconda…

    Go to `Anaconda download page <https://www.anaconda.com/distribution/>`_ or
    `Miniconda download page <https://docs.conda.io/en/latest/miniconda.html>`_.
    Choose your platform and download one of the available installer, *e.g.*, the 3.6 or + version.

    Install the version which you just downloaded, following the instructions on the download page.

    This can take time, go and get your favorite drink...

Installation od spectrochempy
=============================
.. _conda_win:

The following steps have been checked only with windows 10 but should work with previous versions as well.

We highly recommend that all new users install Spectrochempy interface via Conda. You can install Spectrochempy
in a dedicated environment (recommended, steps 4. and 5. below). You can also use your base environment or an
existing environment (then skip steps 4. and 5.)

#.  Open a command prompt (Select the Start button and type cmd), or preferably open the Anaconda Prompt
    in the Anaconda start Menu.

    .. image:: ../../img/Aprompt.png
       :width: 200
       :alt: Anaconda Prompt

#.  Update conda (yes, even if you have just installed the distribution...):

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda update conda

    where `<yourDefaultPath>` is you default workspace directory (often: `C:\\Users\\<user>`)

#.  Add channels to get specific packages:

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda config --add channels conda-forge
        (base) C:\<yourDefaultPath>> conda config --add channels cantera
        (base) C:\<yourDefaultPath>> conda config --add channels spectrocat

#.  Recommended: you can create a dedicated environment. We will name it `scpy` in this
    example

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda env create --name scpy

#.  Recommended: switch to this environment. At this point, `(scpy)` should appear before
    the prompt instead of `(base)`.

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda activate scpy
        (scpy) C:\<yourDefaultPath>>

    .. Note::

        You can make the scipy environment permanent by creating and using the following batch file (.bat)

        .. sourcecode:: bat

            @REM launch a cmd window in scpy environment (path should be adapted)
            @CALL CD C:\<yourWorkingFolder>
            @CALL CMD /K C:\<yourAnacondaFolder>\Scripts\activate.bat scpy

        This script, where `<yourAnacondaFolder>` is the installation directory of your Miniconda/Anaconda distribution
        will open a command prompt  in  C:\\<yourWorkingFolder> with the `scpy` environment activated.

        Save the .bat file, for instance in `C:\\<yourAnacondaFolder>\Scripts\activate-scpy.bat`,
        create a shortcut, name it, for instance, `Anaconda prompt (scpy)` and place it in an easily accessible
        place (e.g. the Windows Startmenu Folder).

#. Install Spectrochempy

    .. sourcecode:: bat

        (scpy) C:\<yourDefaultPath>> conda install spectrochempy

    This can take time, depending on your python installation and the number of missing packages. Go and get your
    favorite drink as they are are downloaded and extracted…

#. Check the installation by running a IPython session by issuing in the terminal
   the following command:

    .. sourcecode:: bash

        (base) C:\<yourDefaultPath>> ipython

    Then execute the following command:

    .. sourcecode:: ipython

        In [1]: from spectrochempy import *

    If this goes well, you should see the following output, indicating that Spectrochempy
    is likely functional !

    .. sourcecode:: ipython

        SpectroChemPy's API - v.0.1.17
        © Copyright 2014-2020 - A.Travert & C.Fernandez @ LCS

Jupyter notebook extensions
===========================

After the installation above, to be able to use spectrochempy in notebooks
with the full plotting capabilities we need to execute the  following command:

.. sourcecode:: bash

    (scpy) C:\<yourDefaultPath>> conda install -c conda-forge widgetsnbextension

Jupyter lab extensions
======================

As for notebooks we need these additional steps (jupytext, jupyterlab-manager and jupyter-matplotlib extensions

.. sourcecode:: bat

    (scpy) C:\<yourDefaultPath>> jupyter nbextension install --py jupytext --user
    (scpy) C:\<yourDefaultPath>> jupyter nbextension enable --py jupytext --user
    (scpy) C:\<yourDefaultPath>> jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

If jupyter lab ask you for building, do it!


Getting started
===============

The recommended next step is to proceed to the |userguide|_


.. _`easy_install`: http://pypi.python.org/pypi/setuptools
.. _`pip`: http://pypi.python.org/pypi/pip

