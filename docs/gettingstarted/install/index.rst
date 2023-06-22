.. _installation:

########################
Installation guide
########################

Prerequisites
*************

`SpectroChemPy` requires a working `Python <http://www.python.org/>`_ installation
(version 3.9 to 3.10).

We highly recommend installing
`Anaconda <https://www.anaconda.com/distribution/>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
distributions which are available for most
platforms and the rest of this guide will mainly use commands for this
distribution.

Miniconda is lighter (400 MB disk space) while Anaconda (3 GB minimum disk space
to download and install)
is much more complete for scientific applications if you intend using python
beyond `SpectroChemPy` . Important packages in Anaconda are also required for `SpectroChemPy`
(e.g., `Matplotib <https://matplotlib.org>`_,
`Numpy <https://numpy.org>`_, `Scipy <https://www.scipy.org>`_,
`Jupyter <https://jupyter.org>`_, …). They are not
included in Miniconda and will be installed anyway when installing `SpectroChemPy` .
So overall, the difference in installation time/disc space won’t be that big
whether you choose Miniconda or Anaconda…

* Go to `Anaconda download page <https://www.anaconda.com/distribution/>`_
  or `Miniconda download page <https://docs.conda.io/en/latest/miniconda.html>`_
  to get one of these distributions.
* Choose your platform and download one of the available installers,
  *e.g.*, the 3.9 or + version (up to now it has been tested upon 3.10).
* Install the version which you just downloaded, following the instructions
  on the download page.

For other python distributions, please check their respective documentation.

Installation of `SpectroChemPy`
*****************************************************

`SpectroChemPy` installation is very similar on the various platforms, except the syntax of some command. We propose here the installation step whether you are on mac/Linux systems, or on Windows.

Additionally it is possible to use a docker container or the Google Colaboratory cloud platform.

* :doc:`install_mac`
* :doc:`install_win`
* :doc:`install_sources`
* :doc:`install_colab`
* :doc:`install_adds`

.. toctree::
    :maxdepth: 3
    :hidden:

    install_mac
    install_win
    install_sources
    install_colab
    install_adds
