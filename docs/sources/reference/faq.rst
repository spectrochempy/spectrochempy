:orphan:

.. _faq:

Frequently asked questions (FAQ)
================================

.. contents:: Table of Contents
   :depth: 2


General
-------
Where are the preference's files saved?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typically, the main application preference file is saved in a hidden directory located in your home user directory:

``$HOME/.spectrochempy/config`` under the name ``spectrochempy_cfg.py``

But if the `SCP_CONFIG_HOME` environment variable is set and the `$SCP_CONFIG_HOME/spectrochempy` directory exists,
it will be that directory.

In principle you should not need to access files in this directory,
but if you want to do it, you can use one of these solutions :

On Mac OSX system, you access to this file by typing in the terminal:

.. sourcecode:: bash

   $ cd ~/.spectrochempy/config
   $ open spectrochempy_cfg.py


On Linux systems, the second command can be replaced by:

.. sourcecode:: bash

   $ nano spectrochempy_cfg.py

or whatever you prefer to read and edit a text file.

Uncomment the line where you want to make a change, for instance change the following line if you want to modify the
default directory where the data are saved:


.. sourcecode:: python

   #c.GeneralPreferences.datadir = ''

to

.. sourcecode:: python

   c.GeneralPreferences.datadir = 'mydatadir/irdata'


Code usage
----------

How to get the index from a coordinate?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The index of a wavelength (or any other type of coord) can be obtained by the `loc2index()` method which will return the index corresponding to the closest value:

.. sourcecode:: python

   >>> X.x.loc2index(2000.0)
   2074


The exact value of the coordinate can the be obtained by:

.. sourcecode:: python

   >>> X.x[2074].values
   1999.53



How to specify a plot with abscissa in ascending or descending order?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default NDDataset with wavenumbers (infrared) or ppm units (NMR) are plotted with coordinates in descending order. This can be prevented by passing `reverse=False`:

.. sourcecode:: python

   X.plot(reverse=False)


Conversely, other plots use by default the ascending order, but this can also be changed:

.. sourcecode:: python

   X.plot(reverse=True)
