.. _install_win:

Installation Guide for Windows
##############################

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

* C/C++ Tools

  Some modules need a C/C++ compilation. They are present by default in Mac OS and Linux platforms. For
  windows, download and install `Build Tools for Visual Studio <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_

  This step is rather long depending on your internet connection speed. Go and get your favorite drink!

Installation
=============
(The following steps has been checked only with windows 10)

.. _conda_win:

Create a new conda environment
******************************

For compatibility issues we STRONGLY recommend using a specific conda environment to use |scpy|.
To do so follows the following steps, depending on your operating system.



#.  Open a command prompt (Select the Start button and type cmd), or preferably open the Anaconda Prompt
    in the Anaconda start Menu.

#.  Update conda (yes, even if you have just installed the distribution...):

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda update conda

    where `<yourDefaultPath>` is you default workspace directory (e.g. `C:\\Users\\<user>`)

#.  Add channels to get specific packages:

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda config --add channels conda-forge

#.  Now we can create the `scpy` environment with all the required python packages.

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> cd spectrochempy
        (base) C:\<yourDefaultPath>> conda env create -f env/scpy.yml

    This also takes time. Go and get second favorite drink, etc... while several python packages are download and
    extraction proceeds...

#.  Switch to this environment. At this point, `(scpy)` should appear before the prompt instead of `(base)`.

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda activate scpy
        (scpy) C:\<yourDefaultPath>>

    Note:

        You can make the scipy environment permanent by creating and using the following batch file (.bat)

        .. sourcecode:: bat

            @REM launch a cmd window in scpy environment (path should be adapted)
            @CALL CD C:\<yourWorkingFolder>
            @CALL CMD /K C:\<yourAnacondaFolder>\Scripts\activate.bat scpy

        This script, where `<yourAnacondaFolder>` is the installation directory of your Miniconda/Anaconda distribution
        will open a command prompt  in  C:\\<yourWorkingFolder> with the `scpy` environment activated.

        Save the .bat file, for instance in `C:\\<yourAnacondaFolder>\Scripts\activate-scpy.bat,
        create a shortcut, name it, for instance, `Anaconda prompt (scpy)` and place it in an easily accessible
        place (e.g. the Windows Startmenu Folder).

Install of the |scpy| package
*****************************

install the |scpy| package in this environment using one of the following method.

Conda install
-------------

Todo

Install from the Bitbucket source repository
--------------------------------------------

Using this method you can install the latest stable version (`MASTER <https://bitbucket.org/spectrocat/spectrochempy/src/master/>`_)

.. sourcecode:: bat

    (scpy) C:\<yourDefaultPath>> pip install https://bitbucket.org/spectrocat/spectrochempy/get/master.zip

or the latest development version (`DEVELOP <https://bitbucket.org/spectrocat/spectrochempy/src/develop/>`_).

This must be done with caution because in this case instabilities are more likely to occurs than
with the (`MASTER <https://bitbucket.org/spectrocat/spectrochempy/src/master/>`_).
It is recommended to use a different conda environnement in this case

.. sourcecode:: bat

    (scpy) C:\<yourDefaultPath>> pip install https://bitbucket.org/spectrocat/spectrochempy/get/develop.zip

Install a developper version (Advanced usage)
---------------------------------------------

Installation of the developper version is described here:  :ref:`develguide`.


Check the Installation
----------------------

Run a IPython session by issuing in the terminal the following command:

.. sourcecode:: bash

    (scpy) C:\<yourDefaultPath>> ipython

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

