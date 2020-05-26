.. _contributing_code:

Contributing to the code
=========================

.. warning::

    UNDER WORK. THIS MAY BE OUTDATED OR WRONG


.. contents:: Table of Contents
   :local:


Summary
---------

To contribute to the development of the source code, you will need to

* to install a working version of python,
* to install the GIT version system if it is not already present in your operating system (*e.g.,* Windows),
* clone |scpy| sources from the remote repository on bitbucket,
* install an IDE, such as PyCharm or Spider, allowing you to code more easily.

Here below are our recommendations for these different steps



Clone the Bitbucket repository
------------------------------

Now that you have git installed, you will be able to clone the Bitbucket repository where the spectrochempy sources
are stored and versioned.

First, it is a good idea to register on the `bitbucket website <https://bitbucket.org/>`_.


Creating a conda environment for development
-----------------------------------------------

In order to separate the devlopment project from a running user |scpy| installation,
it is mandatory to create a new conda environment.

TODO

Required disk space for the development source directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sources for |scpy| in its development version requires ~ 500 Mb of disk space.

If you include a specific python environment (scpy-dev), this will add up ~2.8 Gb in
the miniconda/envs folder.

The master repository was tested on the 3.6 or 3.7 python version.

It will not work with earlier version of python, *i.e.*, < 3.6. No attempt to get such compatibility will be made.


.. _clonescpy:

Install a development version of SpectroChemPy on WINDOWS
-----------------------------------------------------------

#.  Open a command prompt (Select the Start button and type cmd), or preferably open the Anaconda Prompt /
    Powershell Prompt in the Anaconda start Menu.

#.  Update conda (yes, even if you have just installed the distribution...):

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> conda update conda

    where `<yourDefaultPath>` is you default workspace directory (e.g. `C:\\Users\\<user>`)

#.  Add channels to get specific packages:

    .. sourcecode:: bat

        C:\<yourDefaultPath>> conda config --add channels conda-forge
        C:\<yourDefaultPath>> conda config --add channels spectrocat

#.  Check whether `git` is installed :

        (base) C:\<yourDefaultPath>> git --version

    if not, install the `command line version <https://git-scm.com/download/win>`_ of git or the  `command line version + GUI <https://desktop.github.com/>`_


#.  If necessary create your installation directory and go to it.

    We recommend NOT to name it `spectrochempy` because two nested folders `spectrochempy` will also be created at
    the install... you would have then 3 nested `spectrochempy` folders...

    .. sourcecode:: bat

        (base) C:\<yourDefaultPath>> mkdir <yourInstallDirectory>
        (base) C:\<yourDefaultPath>> cd <yourInstallDirectory>

#.  clone spectrochempy in your installation directory:

        (base) C:\<yourInstallDirectory>> git clone https://bitbucket.org/spectrocat/spectrochempy.git

    This may take a while, go and get your favorite drink or whatever else pleases you...

#.  Go in the `spectrochempy` directory and create the scpy-dev environment

    .. sourcecode:: bat

        (base) C:\<yourInstallDirectory>\spectrochempy> cd spectrochempy
        (base) C:\<yourInstallDirectory>\spectrochempy> conda env create -f env/scpy-dev.yml

    This also takes time. Go and get second favorite drink, etc... while the package download and
    extraction proceeds...

#.  Switch to this environment:

    .. sourcecode:: bat

        (base) C:\<yourInstallDirectory>\spectrochempy> activate scpy-dev

#.  At this point, `(scpy-dev)` should appear before the prompt. Then install the spectrochempy package in this environment:

    .. sourcecode:: bat

        (scpy-dev) C:/<your installdir>/spectrochempy> pip install -e .

    Note that you can make the scipy-dev it permanent by creating and using the following batch file (.bat)

    .. sourcecode:: bat

        @REM launch a cmd window in scpy-dev environment (path should be adapted)
        @CALL CD C:<yourWorkingFolder>
        @CALL CMD /K C:<yourAnacondaFolder>\Scripts\activate.bat scpy-dev

    where `<yourWorkingFolder>` is the folder where the prompt window will open (e.g. `users\<username` and
    `<yourAnacondaFolder>` is the Anaconda or Miniconda folder (often `C:\Anaconda3` or `C:\Miniconda3`).
    Save the batch file in e.g. `<yourAnacondaFolder>`, create a shortcut and put it in your desktop or in the
    start menu

#.  If during set up or runtime, some packages with name <pkgname> appear to
    be missing, just install them using

    .. sourcecode:: bat

       (scpy-dev) C:/<your installdir>/spectrochempy> conda install <pkgname>

    or

    .. sourcecode:: bat

       (scpy-dev) C:/<your installdir>/spectrochempy> pip install <pkgname>

#.  Launch python from any working directory:

    .. sourcecode:: bat

        (scpy-dev)  C:\<your workspace>>python

    .. sourcecode:: python

        >>> from spectrochempy import *

    you should then see the following output after few seconds

    ``SpectroChemPy's API - v.0.1a14.dev18+g86dfb85``

    ``(c) Copyright 2014-2020 - A.Travert & C.Fernandez @ LCS``

    and then be able to issue a scpy command:

    .. sourcecode:: python

        >>> NDDataset()

    If this goes well, your install should be fucntional, but not bug-free yet :-(...


Install a development version of SpectroChemPy on MAC OS
---------------------------------------------------------

#.  Git clone the |scpy| `Bitbucket repository <https://bitbucket.org/spectrocat/spectrochempy/src/master/>`_

    .. sourcecode:: bash

       $ git clone git@bitbucket.org:spectrocat/spectrochempy.git <workspace>/spectrochempy

    where `<workspace>` is you programming workspace directory.

    .. note::

       if you want to contribute and push your change to the Bitbucket repository,
       you will need to modify this step. Go fist to :ref:`forkscpy` and then come back here.


#.  Switch to the ``spectrochempy`` directory

    .. sourcecode:: bash

       $ cd <workspace>/spectrochempy


#.  Create a `conda` environment called, for example, **scpy**
    by entering the following commands:

    .. sourcecode:: bash

       $ conda env create -f=env/scpy-dev.yml

    This will add all (or most) of the necessary packages for development.

#.  Switch to this environment:

    .. sourcecode:: bash

        $ conda activate scpy-dev

    You can make it permanent by putting this command in your ``bash_profile``
    (MAC), ``.bashrc`` (LINUX) or using the following batch file (WIN)

    .. sourcecode:: bat

        @REM launch a cmd window in scpy-dev environment (path should be adapted)
        @CALL CD C:\your\favorite\folder
        @CALL CMD /K C:\your\anaconda\folder\Scripts\activate.bat scpy-dev

#. 	Install the spectrochempy package

    Execute the `setup.py` in developper mode

    .. sourcecode:: bash

       $ python setup.py develop

    or use the pip command in developper mode (flag `-e`)

    .. sourcecode:: bash

       $ pip install -e .

#.  If during set up or runtime, some packages with name <pkgname> appear to
    be missing, just install them using

    .. sourcecode:: bash

       $ conda install -n scpy <pkgname>

    ```n scpy`` is just to be sure we install in the correct environment.

.. _forkscpy:

Create a SpectroChemPy fork repository
---------------------------------------

The problem with the above procedure is that you can commit change
made to the application locally, but you won't be able to push any changes to the
``origin`` repository if the maintainer do not give `write` access to it.

To be able to contribute to |scpy|, you will need to create you own **fork** of the
|scpy| repository based on `Bitbucket <https://bitbucket.org/>`. And then from your fork, you can
create pull request to the main repository.

The workflow is the following:

* Create a fork on Bitbucket.
* Clone the forked repository to your local system.
* Modify the local repository.
* Commit your changes.
* Push your changes back to the remote fork on Bitbucket.
* Create a pull request from the forked repository (source) back to the original (destination).

The final step in the workflow is for the maintener of the original repository to merge your changes.

The simplest way is to perform this operation on the `bitbucket.org <https://bitbucket.org/>`_ web site.

* Create an account (if not yet done) or sign in:

  .. image:: images/signin.jpg
     :width: 500 px
     :alt: Sign in on Bitbucket
     :align: center


* Go to the |scpy| repository
  `<https://bitbucket.org/spectrocat/spectrochempy>`_. You should see something like this:

  .. image:: images/scpy_repo.png
     :width: 500 px
     :alt: Spectrochempy repository
     :align: center


* click ``+`` in the sidebar and select `Fork` this repository under `Get to work`.

  .. image:: images/forkit.png
     :width: 500 px
     :alt: Fork
     :align: center


  The system displays the Fork dialog.

  .. image:: images/forkit2.png
     :width: 500 px
     :alt: Fork dialog
     :align: center


* Now you can proceed with the previous installation steps :ref:`clonescpy`. The only change is the
  git command to clone your own |scpy| Bitbucket repository, instead of the official ones.

  .. sourcecode:: bash

     $ git clone git@bitbucket.org:<username>/spectrochempy.git <workspace>/spectrochempy

  where `<username>` is your bitbucket account user name and `<workspace>` is you programming workspace directory.


* After you fork a repository, the original repository is likely to evolve as other users commit changes to it.
  These changes do not appear in your fork automatically. To find out if your fork is missing commits,
  at the bottom of the Repository details card of your fork, you'll see a button with `Sync (# commits behind)`.
  Click this button to pull these commits into your fork.

  .. image:: images/details.png
     :width: 300 px
     :alt: Repository details
     :align: center


Testing SpectroChemPy
---------------------

Tests for SpectroChemPy are executed using
`pytest <https://docs.pytest.org/en/latest/>`_.
It should be present on the system, else install it:

.. sourcecode:: bash

   $ conda install pytest


To run the full suite of tests or only some of them, the best way is to use PyCharm.

However it is possible to execute also the full suite of test, using the following command
from inside the main spectrochempy directory (where the folder ``tests`` resides.

.. sourcecode:: bash

   $ cd <workspace>/spectrochempy/tests
   $ pytest .

Currently it is not possible to use arguments in this command line, as they
will be interpreted by spectrochempy and then produce errors.
To add arguments/options to pytest, use the ``pytest.ini`` file in the ``tests`` folder.



Commit and push to the Bitbucket repository
--------------------------------------------

to do


.. substitutions

.. |git| image:: images/git.png