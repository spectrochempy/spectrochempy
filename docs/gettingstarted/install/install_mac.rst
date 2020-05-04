.. _install_mac:

Installation Guide for Mac and linux
====================================

Installation
-------------

.. _conda_mac:

The following steps have been checked only with OSX catalina but should work with previous versions as well.

We highly recommend that all new users install |scpy| interface via Conda. You can install Spectrochempy
in a dedicated environment (recommended, steps 4. and 5. below). You can also use your base environment or an
existing environment (then skip steps 4. and 5.)

#.  Open a terminal and update conda:

    .. sourcecode:: bash

        (base)  ~ $ conda update conda

    you exact prompt may be different depending on the shell you are using and its configuration

#.  Add channels to get specific packages:

    .. sourcecode:: bash

        (base)  ~ $ conda config --add channels conda-forge
        (base)  ~ $ conda config --add channels cantera
        (base)  ~ $ conda config --add channels spectrocat

#.  Recommended: you can create a dedicated environment. We will name it `scpy` in this
    example

    .. sourcecode:: bash

        (base)  ~ $ conda env create -n scpy

#.  Recommended: switch to this environment. At this point, `(scpy)` should appear before
    the prompt instead of `(base)`.

    .. sourcecode:: bash

        (base)  ~ $ conda activate scpy
        (scpy)  ~ $

    .. Note::

        You can chose to make the `scpy` environment as a default

        Edit the startup profile so that the last line is source activate environment_name.
        In Mac OSX this is ~/.bash_profile. If you use Mac OSX Catalina, it may be ~/.zshrc.
        In linux: this may be ~/.bashrc

        .. sourcecode:: bash

            (scpy)  ~ $ open ~/.bash_profile

        Go to end of file and type the following:

            (scpy)  ~ $ conda activate scpy

        Save and exit File. Start a new terminal window.
        Type the following to see what environment is active

        .. sourcecode:: bash

            (scpy)  ~ $ conda info -e

        The result shows that your are using your environment by default.

#. Install |scpy|

    .. sourcecode:: bash

        (scpy)  ~ $ conda install spectrochempy

    This can take time, depending on your python installation and the number of missing packages.

Install a developper version (Advanced usage)
---------------------------------------------

Installation of the developper version is described here:  :ref:`develguide`.


Check the Installation
-----------------------

Run a `IPython <https://ipython.readthedocs.io/en/stable/>`_ session by issuing in the terminal the following command:

.. sourcecode:: bash

    (scpy) ~ $ ipython

Then execute the following command:

.. sourcecode:: ipython

    In [1]: from spectrochempy import *

If this goes well, you should see the following output, indicating that Spectrochempy
is likely functional !

.. sourcecode:: ipython

    SpectroChemPy's API - v.0.1.17
    © Copyright 2014-2020 - A.Travert & C.Fernandez @ LCS

The recommended next step is to proceed to the :ref:`userguide` or the :ref:`tutorials`

