.. _install_adds:

Install optional dependencies
=============================

Depending on your project, you may complete `SpectroChemPy` with additional but optional dependencies that are only used
for specific methods.
For example, :mod:`cantera_utilities.py` requires the ``cantera`` package.

If the optional dependency is not installed, SpectroChempy will raise an ``ImportError`` when
the method requiring that dependency is called.

Examples and test data
----------------------

When installing the `SpectroChemPy` package, the data and examples used in documentation and for testing are not provided.
If you want to try the documentation examples, you need to install them separately using:

.. sourcecode:: bash

   $ mamba install -c spectrocat spectrochempy_data


Alternatively you can download an archive on `github <https://github.com/spectrochempy/spectrochempy_data/tags>`__
and unpack it in the directory of your choice. In this case you may need to adapt the path for the reading functions.


Cantera
-------

Cantera is a suite of tools for problems involving chemical kinetics, thermodynamics and transport process
(see `cantera documentation <https://cantera.org>`__).

We provide some functionalities based on cantera in `SpectroChemPy` . If you want to use it you need first to install cantera:

.. sourcecode:: bash

   $ mamba install -c cantera cantera

for the stable version or

.. sourcecode:: bash

   $ mamba install -c cantera/label/dev cantera

for the development version.


QT
--

If you like to have the matplotlib qt backend for your plots, you need to install the pyqt library.

.. sourcecode:: bash

   $ mamba install pyqt

Then you can use *e.g.,* the qt backend in notebooks using the IPython "magic" line:

.. sourcecode:: ipython

   %matplotlib qt
