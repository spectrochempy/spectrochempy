.. _install_sources:

Installation from sources (master or develop versions)
======================================================

Installing git
---------------

To install |scpy| from sources, you first need to install ``git`` on your system if it is not already present.

Git is a free and open source distributed control system used in well-known software repositories, such as
`GitHub <https://github.com>`__ or `Bitbucket <https://bitbucket.org>`__. For this project, we use a GitHub
repository: `spectrochempy repository <https://github.com/spectrochempy/spectrochempy>`__.

Depending on your operating system, you may refer to these pages for installation instructions:

-  `Download Git for macOS <https://git-scm.com/download/mac>`__ (One trivial option is to install
   `XCode <https://developer.apple.com/xcode/>`__ which is shipped with the git system).

-  `Download Git for Windows <https://git-scm.com/download/win>`__.

-  `Download for Linux and Unix <https://git-scm.com/download/linux>`__.
   For the common Debian/Ubuntu distribution, it is as simple as typing in the Terminal:

   .. sourcecode:: bash

      $ sudo apt-get install git

-  Alternatively, once miniconda or anaconda is installed (see :ref:`installation` or below if it not yet done),
   one can use conda to install git:

   .. sourcecode:: bash

      $ conda install git

To check whether or not *git* is correctly installed, use this command in the terminal:

.. sourcecode:: bash

   $ git --version

Cloning the repository locally
-------------------------------

The fastest way is to type these commands in a terminal on your machine:

.. sourcecode:: bash

   $ git clone -depth=50 https://github.com/spectrochempy/spectrochempy.git
   $ cd spectrochempy
   $ git remote add upstream https://github.com/spectrochempy/spectrochempy.git

These commands create the directory ``spectrochempy`` and connects your repository to the ``upstream`` (master branch) |scpy| repository.

.. _installing_conda:

Create a conda environment
--------------------------

* Install either `Anaconda <https://www.anaconda.com/download/>`_, `miniconda
  <https://conda.io/miniconda.html>`_, or `miniforge <https://github.com/conda-forge/miniforge>`_
* Make sure your conda is up to date (``conda update conda``)


* ``cd`` to the |scpy| source directory (*i.e.,* ``spectrochempy`` created previously)


* Generate an environment file using the script ``env_create.py``:

  .. sourcecode:: bash

     $ python .ci/env/env_create.py -v 3.9 --dev scpy_dev.yml

  This will create the ``scpy_dev.yml`` file that we will use in the next step.

* Create and activate the environment using python 3.9. This will create a new environment and will not touch
  any of your other existing environments, nor any existing Python installation.
  (conda installer is somewhat very slow, this is why we prefer to replace it by `mamba <https://https://github.com/mamba-org/mamba>`__

  .. sourcecode:: bash

     $ conda update conda
     $ conda install -c conda-forge mamba 
     $ mamba env create -n scpy_dev -f scpy_dev.yml
     $ conda activate scpy_dev

  If you prefer to work with ``3.7`` or ``3.8`` python version, you just change ``-v 3.9`` by ``-v 3.8`` or ``3.7``.

Install |scpy| in this environment
----------------------------------

.. sourcecode:: bash

   $ python -m pip install .


At this point you should be able to ``import spectrochempy``:

.. sourcecode:: bash

   (scpy_dev) $ python


This start an interpreter in which you can check your installation.

.. sourcecode:: python

   >>> import spectrochempy as scp
   >>> print(scp.version)
   SpectroChemPy's API ...
   >>> exit()

To view your environments:

.. sourcecode:: bash

   conda env list

To return to the base environment:

.. sourcecode:: bash

   conda deactivate

Updating |scpy|
---------------

One definite advantage of installing for git sources is that you can update your version very easily.

To update your local master branch, you can do:

.. sourcecode:: bash

    git pull upstream master --ff-only

and if some changes are notified, run pip install again:

.. sourcecode:: bash

    python -m pip install .


To go further and eventually contribute to the code on the upstream, you can consult the :ref:`develguide`.
