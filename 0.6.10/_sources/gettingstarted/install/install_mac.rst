.. _install_mac:

Installation Guide for Mac OSX
===============================

Installation using Conda
------------------------

.. _conda_mac:

The following steps have been checked only with OSX-Catalina but should work
with previous versions as well and
hopefully on the more recent OSX version.

As said before, we highly recommend that all new users install `SpectroChemPy`
interface via Conda. You can install `SpectroChemPy` in a dedicated
environment (recommended, steps 4. and 5. below). You can also use your base
environment or an existing environment
(then skip steps 4. and 5.)

#.  Open a terminal and update conda:

    .. sourcecode:: bash

        (base)  ~ $ conda update conda

    your exact prompt may be different depending on the shell you are using and
    its configuration

#.  Add channels to the base configuration to simplify the installation of
    specific packages from different sources:

    .. sourcecode:: bash

        (base)  ~ $ conda config --add channels spectrocat
        (base)  ~ $ conda config --add channels conda-forge
        (base)  ~ $ conda config --add channels cantera

    Note that the last line about cantera is only require if you intend to work
    the kinetics modules of `SpectroChemPy` .

#.  **Recommended**: you should create a dedicated environment in order to
    isolate the changes made on the installed library from any other previous
    installation for another application.

    We will name this new environment ``scpy`` in this example
    but of course you can use whatever name you want.

    .. sourcecode:: bash

        (base)  ~ $ conda create -n scpy

    Switch to this environment. At this point, `(scpy)` should
    appear before the prompt instead of `(base)` .

    .. sourcecode:: bash

        (base)  ~ $ conda activate scpy
        (scpy)  ~ $

    .. Note::

       You can chose to make the `scpy` environment as a default

       Edit the startup profile so that the last line is source activate
       environment_name.

       In Mac OSX this is ~/.bash_profile. If you use Mac OSX Catalina, it may be
       ~/.zshrc.

       In linux: this may be ~/.bashrc

       .. sourcecode:: bash

            (scpy)  ~ $ open ~/.bash_profile

       Go to end of file and type the following:

       .. sourcecode:: bash

            (scpy)  ~ $ conda activate scpy

       Save and exit File. Start a new terminal window.
       Type the following to see what environment is active

       .. sourcecode:: bash

            (scpy)  ~ $ conda info -e

       The result shows that your are using your environment by default.

#. Install `SpectroChemPy`

   The conda installer has to solve all packages dependencies and is definitely
   a bit slow. So we recommend to install `mamba <https://github.com/mamba-org/mamba>`__
   as a drop-in replacement via:

   .. sourcecode:: bash

        (scpy)  ~ $ conda install mamba

   To install a stable version of spectrochempy, then you just have to do :

   .. sourcecode:: bash

        (scpy)  ~ $ mamba install spectrochempy

   or if you rather prefer not to use mamba:

   .. sourcecode:: bash

        (scpy)  ~ $ conda install spectrochempy


   This can take time, depending on your python installation and the number of
   missing packages.

   If you prefer to deal with the latest development version, you must use the
   following command to install from the
   `spectrocat/label/dev <https://anaconda.org/spectrocat/spectrochempy>`_
   channel instead of the `spectrocat` channel:

   .. sourcecode:: bash

        (scpy)  ~ $ mamba install -c spectrocat/label/dev spectrochempy

Installation using pip
----------------------

If you prefer to use pip, here are the installation steps. We assume that you have a working installation of python > 3.6.

#. Open a terminal and update pip:

   .. sourcecode:: bash

        $ python -m pip install --user --upgrade pip

#. Creating a virtual environment

   .. sourcecode:: bash

        $ python -m venv env
        $ source env/bin/activate

   Check that you in the correct environment

   .. sourcecode:: bash

       (env) $ which python

       .../env/bin/python

#. Install all required packages

   The easiest way to achieve this is to use the requirements.txt present on our github repository or in the present documentation (<link>)

   .. sourcecode:: bash

     (env) $ python -m pip install -r https://www.spectrochempy.fr/downloads/requirements.txt

#. Install spectrochempy from pypi

   .. sourcecode:: bash

       (env) $ python -m pip install spectrochempy


Check the Installation
-----------------------

Run a `IPython <https://ipython.readthedocs.io/en/stable/>`_ session by issuing
in the terminal the following command:

.. sourcecode:: bash

    (scpy) ~ $ ipython

Then execute the following command:

.. sourcecode:: ipython

    In [1]: from spectrochempy import *

If this goes well, you should see the following output, indicating that
Spectrochempy is likely functional !

.. sourcecode:: ipython

    SpectroChemPy's API - v.0.1.17
    © Copyright 2014-2020 - A.Travert & C.Fernandez @ LCS

The recommended next step is to proceed to the :ref:`userguide` .
