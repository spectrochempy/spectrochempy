.. _install_adds:

Install additional libraries
=============================

Depending on your project, you may complete |scpy| with additional features.

Cantera
-------

Cantera is a suite of tools for problems involving chemical kinetics, thermodynamics and transport process
(see `cantera documentation <https://cantera.org>`__).

We provide some functionalities based on cantera in |scpy|. If you want to use it you need first to install cantera:

.. sourcecode:: bash

   $ mamba install -c cantera cantera

for the stable version or

.. sourcecode:: bash

   $ mamba install -c cantera/label/dev cantera

for the development version.


QT
--

IF you like to have the matplotlib qt backend for your plots, you need to install the pyqt library.

.. sourcecode:: bash

   $ mamba install pyqt

Then you can use *e.g.,* the qt backend in notebooks using the IPython "magic" line:

.. sourcecode:: ipython

   %matplotlib qt
