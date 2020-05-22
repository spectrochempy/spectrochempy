.. _install_dev:

Install a the latest development version
========================================

Install Miniconda
------------------

First, if it was not yet done, install miniconda.

Although `Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_ is certainly the
best choice to install python and the many packages
needed for scientific development, it has some disadvantages for developers:
The installed packages are so numerous that it becomes difficult to know which ones
are really necessary for the project.

That's why we will turn to
`Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_ instead, in order to
have a better control and a better detection of possible incompatibilities.

*  To install Miniconda:

   Go to the `Miniconda download page <https://docs.conda.io/en/latest/miniconda.html>`_.
   Choose your platform and download one of the available installer, *e.g.*, the 3.7 or + version.

*  Install the version which you just downloaded, following the instructions on the download page.

   Look `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#choosing-a-version-of-anaconda-or-miniconda>`_
   for more information.

Install git
---------------

|git| `Git <https://git-scm.com>`_ is a free and open source distributed control system used in well-known software
repositories, such as `GitHub <https://github.com>`_ or `Bitbucket <https://bitbucket.org>`_.  For this project, we use
a `GitHub <https://github.com>`_ repository.

Depending on your operating system you may refer to these pages for installation instructions:

* `Download Git for macOS <https://git-scm.com/download/mac>`_ (One trivial option is to install
  `XCode <https://developer.apple.com/xcode/>`_ which is shipped with the git system).

* `Download Git for Windows <https://git-scm.com/download/win>`_.

* `Download for Linux and Unix <https://git-scm.com/download/linux>`_. For the common Debian/Ubuntu distribution,
  it is as simple as typing in the Terminal:

  .. sourcecode:: bash

        sudo apt-get install git

* Alternatively, once miniconda or anaconda is installed, one can use conda to install git:

  .. sourcecode:: bash

        conda install git

To check whether or not *git* is correctly installed, use:

.. sourcecode:: bash

    git --version

Optional: install a GUI git client
------------------------------------

Once your installation of **git** is complete, it may be useful (and we recommend it) to install a GUI client for the git
version system.

We use the `SourceTree client <https://www.sourcetreeapp.com>`_
(which can be installed on both Windows and Mac operating systems).
To configure and learn how to use the sourcetree GUI application, you can consult
this `tutorial <https://confluence.atlassian.com/bitbucket/tutorial-learn-bitbucket-with-sourcetree-760120235.html>`_

However, any other GUI can be interesting, or if you prefer, it is possible to use only the command line in a terminal.

Note that an IDE such as PyCharm (see ...) have an integrated GUI git client which can be used in place of an external application.