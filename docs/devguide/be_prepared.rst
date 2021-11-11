.. _be_prepared:

Be prepared to work on the code
================================

To contribute further to the code and documentation, you will need learning how to work with GitHub and the |scpy| code base.

.. _contributing.version_control:

Version control, Git, and GitHub
---------------------------------

The code of |scpy| is hosted on `GitHub <https://www.github.com/spectrochempy/spectrochempy>`__. To contribute, you will need to register for a `free GitHub account <https://github.com/signup/free>`__. Actually, you may have already done this while submitting issues, so you are also almost ready to contribute to the code.

The reason we use `Git <https://git-scm.com/>`__ to version controlling our code is to allow several people to work on this project simultaneously.

However, working with `Git <https://git-scm.com/>`__ is unfortunately not the easiest step for a newcomer when trying to contribute to an open-source software. But learning this versioning system at the same time as developing with Python can be very rewarding for your daily work.

To learn `Git <https://git-scm.com/>`__, you may want to check out the `GitHub help pages <https://help.github.com/>`_ or the
`NumPy's documentation <https://numpy.org/doc/stable/dev/index.html>`__. There is no shortage of resources on the web on this subject and many tutorials are available.

Below we give some essential information to get you started with **Git**, but of course if you encounter difficulties, don't hesitate to ask for help. This may help to solve your problems faster, but it also allows us to improve this part of our documentation based on your feedback.

Installing git
---------------

`GitHub <https://help.github.com/set-up-git-redirect>`__ has instructions for installing and configuring git.  All these steps need to be completed before you can work seamlessly between your local repository and GitHub.

Git is a free and open source distributed control system used in well-known software repositories, such as
`GitHub <https://github.com>`__ or `Bitbucket <https://bitbucket.org>`__. For this project, we use a GitHub
repository: `spectrochempy repository <https://github.com/spectrochempy/spectrochempy>`__.

Depending on your operating system, you may refer to these pages for installation instructions:

-  `Download Git for macOS <https://git-scm.com/download/mac>`__ (One trivial option is to install `XCode <https://developer.apple.com/xcode/>`__ which is shipped with the git system).

-  `Download Git for Windows <https://git-scm.com/download/win>`__.

-  `Download for Linux and Unix <https://git-scm.com/download/linux>`__. For the common Debian/Ubuntu distribution, it is as simple as typing in the Terminal:

   .. sourcecode:: bash

       sudo apt-get install git

-  Alternatively, once miniconda or anaconda is installed (see :ref:`contributing.dev_env`), one can use conda to install
   git:

   .. sourcecode:: bash

       conda install git

To check whether or not *git* is correctly installed, use

.. sourcecode:: bash

   git --version

Optional: installing a GUI git client
-------------------------------------

Once your installation of git is complete, it may be useful to install a GUI client for the git version system.

We have been using `SourceTree client <https://www.sourcetreeapp.com>`__ (which can be installed on both Windows and Mac operating systems). To configure and learn how to use the sourcetree GUI application, you can consult this
`tutorial <https://confluence.atlassian.com/bitbucket/tutorial-learn-bitbucket-with-sourcetree-760120235.html>`__

However, any other GUI can be interesting such as `Github-desktop <https://desktop.github.com>`__, or if you prefer, you can stay using the command line in a terminal.

.. note::

   Many IDE such as `PyCharm <https://www.jetbrains.com/fr-fr/pycharm/>`__ have an integrated GUI git client which can be used in place of an external application. This is an option that we use a lot (in combination to the more visual SourceTree application)

.. _contributing.forking:

Forking the spectrochempy repository
------------------------------------

You will need your own fork to work on the code. Go to the `SpectroChemPy project page <https://github.com/spectrochempy/spectrochempy>`__ and hit the ``Fork`` button to create an exact copy of the project on your account.

Then you will need to clone your fork to your machine. The fastest way is to type these commands in a terminal on your machine:

.. sourcecode:: bash

   git clone https://github.com/your-user-name/spectrochempy.git localfolder
   cd localfolder
   git remote add upstream https://github.com/spectrochempy/spectrochempy.git

This creates the directory ``localfolder`` and connects your repository to the upstream (main project) |scpy| repository.

.. _contributing.dev_env:

Creating a Python development environment
------------------------------------------

To test out code and documentation changes, you'll need to build |scpy| from source, which requires a Python environment.

* Install either `Anaconda <https://www.anaconda.com/download/>`_, `miniconda
  <https://conda.io/miniconda.html>`_, or `miniforge <https://github.com/conda-forge/miniforge>`_
* Make sure your conda is up to date (``conda update conda``)
* Make sure that you have :ref:`cloned the repository <contributing.forking>`

* ``cd`` to the |scpy| source directory (*i.e.,* ``localfolder`` created previously)

We'll now install |scpy| in development mode following 2 steps:

1. Create and activate the environment. This will create a new environment and will not touch
   any of your other existing environments, nor any existing Python installation.
   (conda installer is somewhat very slow, this is why we prefer to replace it by
   `mamba <https://https://github.com/mamba-org/mamba>`__.

   .. sourcecode:: bash

      conda update conda -y
      conda config --add channels conda-forge
      conda config --add channels cantera
      conda config --add channels spectrocat
      conda config --set channel_priority flexible
      conda install mamba jinja2

   Here we will create un environment using python in its version 3.9
   but it is up to you to install any version from 3.6.9 to 3.9.
   Just change the relevant information in the code below (the first line use a
   script to create the necessary yaml
   file containing all information about the packages to install):

   .. sourcecode:: bash

      python .ci/env/env_create.py -v 3.9 --dev scpy3.9.yml
      mamba env create -f .ci/env/scpy3.9.yml
      conda activate scpy3.9

2. Install |scpy|

   Once your environment is created and activated, we must install SpectroChemPy
   in development mode.

   .. sourcecode:: bash

      (scpy3.9) $ cd <spectrochempy folder>
      (scpy3.9) $ python setup.py develop

   At this point you should be able to import spectrochempy from your local
   development version:

   .. sourcecode:: bash

      (scpy3.9) $ python

   This start an interpreter in which you can check your installation

   .. sourcecode:: python

     >>> import spectrochempy as scp
     >>> print(scp.version)
     SpectroChemPy's API ...
     >>> exit()

Controling the environments
---------------------------

You can create as many environment you want, using the method above
(for example with different versions of python)

To view your environments:

.. sourcecode:: bash

   conda info -e

To return to your root environment:

.. sourcecode:: bash

   conda deactivate

See the full conda docs `here <https://conda.pydata.org/docs>`__.

Creating a branch
-----------------

Generally we want the master branch to reflect only production-ready code, so you will have create a
feature branch for making your changes. For example:

.. sourcecode:: bash

    git branch my_new_feature
    git checkout my_new_feature

The above can be simplified to:

.. sourcecode:: bash

    git checkout -b my_new_feature

This changes your working directory to the ``my-new-feature`` branch.  Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to spectrochempy. You can have many ``my-other-new-feature``
branches and switch in between them using the:

.. sourcecode:: bash

    git checkout command.

When creating this branch, make sure your master branch is up to date with the latest upstream master version. To update your local master branch, you can do:

.. sourcecode:: bash

    git checkout master
    git pull upstream master --ff-only

.. When you want to update the feature branch with changes in master after you
.. created the branch, check the section on :ref:`updating a PR
<contributing.update-pr>`.
