.. _install_win:

Installation Guide for Windows
===============================

Installation
-------------

.. _conda_win:

The following steps have been checked only with windows 10 but should work with
previous versions as well.

We highly recommend that all new users install |scpy| interface via Conda. You
can install Spectrochempy in a dedicated environment (recommended, steps 4. and
5. below). You can also use your base environment or an existing environment
(then skip steps 4. and 5.)

#.  Open a command prompt (Select the Start button and type cmd), or preferably
    open the Anaconda Prompt in the Anaconda start Menu.

    .. image:: images/prompt.png
       :width: 200
       :alt: Anaconda Prompt


#.  Update conda (yes, even if you have just installed the distribution...):

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda update conda

    where `<yourDefaultPath>` is you default workspace directory (often:
    `C:\\Users\\<user>`)

#.  Add channels to get specific packages:

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda config --add channels conda-forge
        (base) C:\<yourDefaultPath>> conda config --add channels spectrocat
        (base) C:\<yourDefaultPath>> conda config --add channels cantera

#.  **Recommended**: you can create a dedicated environment. We will name it
    `scpy` in this example but of course you can use whatever name you want.

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda create --name scpy

    Switch to this environment. At this point, `(scpy)` should
    appear before the prompt instead of `(base)`.

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda activate scpy
        (scpy) C:\<yourDefaultPath>>

    .. Note::

        You can make the scipy environment permanent by creating and using the
        following batch file (.bat)

        .. sourcecode:: bat

            @REM launch a cmd window in scpy environment (path should beadapted)
            @CALL CD C:\<yourWorkingFolder>
            @CALL CMD /K C:\<yourAnacondaFolder>\Scripts\activate.bat scpy

        This script, where `<yourAnacondaFolder>` is the installation directory
        of your Miniconda/Anaconda distribution
        will open a command prompt  in  C:\\<yourWorkingFolder> with the `scpy`
        environment activated.

        Save the .bat file, for instance in
        `C:\\<yourAnacondaFolder>\Scripts\activate-scpy.bat`,
        create a shortcut, name it, for instance, `Anaconda prompt (scpy)`
        and place it in an easily accessible place (e.g. the Windows Startmenu
        Folder).

#. Install |scpy|

   The conda installer has to solve all packages dependencies and is definitely
   a bit slow. So we recommand to install `mamba <https://github.com/mamba-org/mamba>`__
   as a drop-in replacement via:

   .. sourcecode:: bash

        (scpy) C:\<yourDefaultPath>> conda install mamba

   To install a stable version of spectrochempy, then you just have to do :

   .. sourcecode:: bash

        (scpy) C:\<yourDefaultPath>> mamba install spectrochempy

   or if you rather prefer not to use mamba:

   .. sourcecode:: bat

        (scpy) C:\<yourDefaultPath>> conda install spectrochempy

   This can take time, depending on your python installation and the number of
   missing packages.


   If you prefer to deal with the latest development version, you must use the
   following command to install from the
   `spectrocat/label/dev <https://anaconda.org/spectrocat/spectrochempy>`_
   channel instead of the `spectrocat` channel:

   .. sourcecode:: bat

        (scpy) C:\<yourDefaultPath>> mamba install -c spectrocat/label/dev spectrochempy


Check the Installation
------------------------

Check the installation by running a
`IPython <https://ipython.readthedocs.io/en/stable/>`_ session by issuing in
the terminal the following command:

.. sourcecode:: bat

    (scpy) C:\<yourDefaultPath>> ipython

Then execute the following command:

.. sourcecode:: ipython

    In [1]: from spectrochempy import *

If this goes well, you should see the following output, indicating that
Spectrochempy is likely functional !

.. sourcecode:: ipython

    SpectroChemPy's API - v.0.1.17
    © Copyright 2014-2020 - A.Travert & C.Fernandez @ LCS


The recommended next step is to proceed to the :ref:`userguide`.
