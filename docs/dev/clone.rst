
Fork and clone our GitHub repository
=====================================

Before cloning the SpectroChemPy GitHub directory, we need a working git installation.

## Install git

[Git](https://git-scm.com) is a free and open source distributed control system used in well-known software
repositories, such as [GitHub](https://github.com) or [Bitbucket](https://bitbucket.org).  For this project, we use a [GitHub](https://github.com) repository.

Depending on your operating system you may refer to these pages for installation instructions:

* [Download Git for macOS](https://git-scm.com/download/mac) (One trivial option is to install
  [Xcode](https://developer.apple.com/xcode/) which is shipped with the git system).

* [Download Git for Windows](https://git-scm.com/download/win).

* [Download for Linux and Unix](https://git-scm.com/download/linux). For the common Debian/Ubuntu distribution,
  it is as simple as typing in the Terminal:

        sudo apt-get install git

* Alternatively, once miniconda or anaconda is installed (see [[Install-and-configure-miniconda]]), one can use conda to install git:

        conda install git

To check whether or not *git* is correctly installed, use:

    git --version

## Optional: install a GUI git client

Once your installation of **git** is complete, it may be useful (and we recommend it) to install a GUI client for the git version system.

We have been using [SourceTree client](https://www.sourcetreeapp.com)
(which can be installed on both Windows and Mac operating systems).
To configure and learn how to use the sourcetree GUI application, you can consult
this [tutorial](https://confluence.atlassian.com/bitbucket/tutorial-learn-bitbucket-with-sourcetree-760120235.html)

However, any other GUI can be interesting, or if you prefer, it is possible to use only the command line in a terminal.

Note that an IDE such as PyCharm (see ...) have an integrated GUI git client which can be used in place of an external application. This is an option that we use a lot (in combination to the more visual SourceTree application)
