.. _install:

Installation Guide
###################

.. contents::
	:local:

Where to Get |scp|
==================

To get a working installation of |scp| , on any platform (windows, mac OS X, Linux ... ),
several solutions are (or will be soon) available.

* :ref:`binaries`

* :ref:`pypi`

* :ref:`dev`

* :ref:`clone`


Requirements
============

|scp| requires a working python installation.

* `Python <http://www.python.org/>`_

Currently, only the python version 3.5 has been tested.

The following libraries are also required:

* `Numpy <http://numpy.scipy.org>`_

* `Scipy <http://www.scipy.org/>`_

* `Matplotlib <http://matplotlib.sourceforge.net/index.html>`_

* `Traits <http://code.enthought.com/projects/traits/>`_


Follow the instructions to install these packages on those sites, or, far easier,
install them as packages from your operating system
(e.g. apt-get or the synaptic GUI on Ubuntu, `Macports <http://www.macports.org/>`_ on OS X, etc.).

Regarding the installation of all these above packages, we highy recommend to install EPD python framework (a much straitforward solution!)
which is available for most platforms:

#. install the **Anaconda Scientific Python Distribution** :
Go to `http://continuum.io/downloads <http://continuum.io/downloads>`_ and follow the instructions for your platform -
and if you register as academic member of the university you get interesting add-ons

or

#. install the **Canopy Enthought Distribution** :
Go to `https://store.enthought.com/downloads/ <https://store.enthought.com/downloads/>`_
(it's a commercial distribution, but you have a free version Canopy Express -
and if you register with your academic email, you can get the full academic version)


Finally, after installing your python distribution, you can use the `pip`_
installer for a really easy installation of |scp| from the
`Pypi (spectrochempy) <https://pypi.python.org/pypi/spectrochempy>`_ repository.


.. TODO::

   Conda installer


Installation
=============

.. _binaries:

Installation from Binaries
**************************

Not yet available, sorry.


.. _pypi:

Standard Installation from PyPi sources
***************************************

Very simple, use the following command in a terminal:

.. code-block:: bash

    $ pip install spectrochempy

or to update a previous installation with the latest stable release:

.. code-block:: bash

    $ pip install -U spectrochempy

.. _dev:

Installation from Developpement Sources
***************************************

.. warning::

   These sources may be unstable or even broken.


Downloads zip/tar archives working for all platforms are available.

	* `tar archives <xxx>`_

	* `zip archives <xxx>`_

or on PyPi:

	* `Download tar.gz archives from PyPi <http://pypi.python.org/pypi/spectrochempy>`_

Ungzip and untar the source package, ** *cd to the new directory* **, and execute:

.. code-block:: bash

    $ pip install .

or better :

.. code-block:: bash

	$ pip install -e .

to install it in the developper mode.

.. tip::

	On most UNIX-like systems, you’ll probably need to run these commands as
	root or using sudo.

.. _clone:

Clone or Fork of the Bitbucket Repository
*****************************************

Alternatively, you can make a clone/fork of the github sources at:

* `https://bitbucket.org/spectrocat/spectrochempy  <https://bitbucket.org/spectrocat/spectrochempy>`_

This is the recommended solution for developpers
and those who would like to contribute


Check the Installation
======================

Run a IPython session by issuing in the terminal the following command::

	$ ipython

Then execute two commands as following:

.. sourcecode:: ipython

    In [1]: from spectrochempy.api import *

    In [2]: NDDataset()

If this goes well, the |scp| application is likely functional.

Getting started
===============

The recommended next step is to proceed to the :ref:`userguide`


.. _`easy_install`: http://pypi.python.org/pypi/setuptools
.. _`pip`: http://pypi.python.org/pypi/pip
