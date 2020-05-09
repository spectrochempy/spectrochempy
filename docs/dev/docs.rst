.. _compiling_docs:

.. warning::

    **THIS IS OUTDATED AND NEED TO BE CORRECTED**


.. contents:: Table of Contents
   :local:

Compiling the docs
======================

To build the doc, we need the following packages:

* sphinx
* nbsphinx, to convert notebook to sphinx pages
* sphinx-gallery, to convert python \*.py files to examples for the gallery.
* sphinx-nbexamples, to convert \*.ipynb notebooks into example for the gallery

These packages are available on conda-forge or pypi. They should have been installed during the previous steps.

Assuming you are in the main spectrochempy directory,
to rebuild the doc, just do:

.. sourcecode:: bash

   $cd docs
   $python builddocs.py clean html

or to update it after some changes:

.. sourcecode:: bash

   $cd docs
   $python builddocs.py html

The generated file are located in a directory (spectrochempy_doc) at the same level as the spectrochempy directory.

To display the documentation (on mac. For window the command `start` should work or something equivalent on linux):

.. sourcecode:: bash

   $cd ../../spectrochempy_doc/html
   $open index.html

you can also double-click on the index.html file in your file explorer (may be simpler!).