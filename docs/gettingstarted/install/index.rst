.. _installation:

########################
Installation guide
########################

Prerequisites
*************

|scpy| requires a working `Python <http://www.python.org/>`_ installation
(version 3.6.9 or higher).

We highly recommend to install
`Anaconda <https://www.anaconda.com/distribution/>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
distributions which are available for most
platforms and the rest of this guide will mainly use commands for this
distribution.

Miniconda is lighter (400 MB disk space) while Anaconda (3 GB minimum disk space
to download and install)
is much more complete for scientific applications if you intend using python
beyond |scpy|. Important packages in Anaconda are also required for |scpy|
(e.g., `Matplotib <https://matplotlib.org>`_,
`Numpy <https://numpy.org>`_, `Scipy <https://www.scipy.org>`_,
`Jupyter <https://jupyter.org>`_, …). They are not
included in Miniconda and will be installed anyway when installing |scpy|.
So overall, the difference in installation time/disc space won’t be that big
whether you choose Miniconda or Anaconda…

* Go to `Anaconda download page <https://www.anaconda.com/distribution/>`_
  or `Miniconda download page <https://docs.conda.io/en/latest/miniconda.html>`_
  to get one of these distribution.
* Choose your platform and download one of the available installer,
  *e.g.*, the 3.7 or + version (up to now it has been tested upon 3.9,
  but it is likely to work with upper version numbers).
* Install the version which you just downloaded, following the instructions
  on the download page.

Installation of |scpy|
*****************************************************

|scpy| installation is very similar on the various platform, except the syntax
of some command. We propose here the installation step whether you are on
mac/linux systems, or on Windows. Additionaly it is possible to use a docker
container or the Google Colaboratory cloud platform.

* :doc:`install_mac`
* :doc:`install_win`
* :doc:`install_docker`
* :doc:`install_sources`
* :doc:`install_colab`
* :doc:`install_adds`

.. toctree::
    :maxdepth: 3
    :hidden:

    install_mac
    install_win
    install_docker
    install_sources
    install_colab
    install_adds
