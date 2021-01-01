:orphan:

.. _home:

#####################################################################
Processing, analysing and modelling spectroscopic data
#####################################################################

.. toctree::
    :hidden:
    :caption: Table of Content


.. toctree::
    :hidden:
    :caption: Summary


|scpy| is a framework for processing, analyzing and modeling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. It is a cross platform software,
running on Linux, Windows or OS X.

Among its major features:

#.  Import data from experiments or modeling programs with their *metatdata*
    (title, units, coordinates, ...)
#.  Preprocess these data: baseline correction, (automatic) subtraction,
    smoothing, apodization...
#.  Manipulate single or multiple datasets: concatenation, splitting, alignment
    along given dimensions, ...
#.  Explore data with exploratory analyses methods such as `SVD`, `PCA`, `EFA`
    and visualization capabilities ...
#.  Modelling single or multiple datasets with curve fitting / curve modelling
    (`MCR-ALS`) methods...
#.  Export data and analyses to various formats: `csv`, `xls`, `JCAMP-DX`,  ...
#.  Embed the complete workflow from raw data import to final analyses in a
    Project Manager

.. only:: html

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/version.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/platforms.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/latest_release_date.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

.. warning::

    **This software is not yet publicly released**.

    |scpy| is still experimental and under active development.
    Its current design is subject to major changes, reorganizations,
    bugs and crashes!!! Please report any issues to the
    `Issue Tracker  <https://redmine.spectrochempy.fr/projects/spectrochempy/issues>`_

.. toctree::
   :hidden:

   Home <self>

****************
Getting Started
****************

* :doc:`gettingstarted/whyscpy`
* :doc:`gettingstarted/overview`
* :doc:`gettingstarted/gallery/auto_examples/index`
* :doc:`gettingstarted/install/index`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    gettingstarted/whyscpy
    gettingstarted/overview
    Examples <gettingstarted/gallery/auto_examples/index>
    Installation <gettingstarted/install/index>


.. _userguide:


***********************
User's Guide
***********************

* :doc:`userguide/introduction/index`
* :doc:`userguide/objects`
* :doc:`userguide/units/index`
* :doc:`userguide/import_export/index`
* :doc:`userguide/plotting/index`
* :doc:`userguide/processing/index`
* :doc:`userguide/analysis/index`
* :doc:`userguide/databases/index`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User's Guide

    userguide/introduction/index
    userguide/objects
    userguide/units/index
    userguide/import_export/index
    userguide/plotting/index
    userguide/processing/index
    userguide/analysis/index
    userguide/databases/index

***********************
Reference & Help
***********************

* :doc:`userguide/reference/changelog`
* :doc:`userguide/reference/index`
* :doc:`userguide/reference/preference`
* :doc:`userguide/reference/faq`
* :doc:`devguide/issues`
* :doc:`devguide/examples`
* :doc:`devguide/contributing`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference & Help

    userguide/reference/changelog
    userguide/reference/index
    userguide/reference/preference
    userguide/reference/faq
    Bug reports & feature request <devguide/issues>
    Sharing examples <devguide/examples>
    devguide/contributing


********
Credits
********

* :doc:`credits/credits`
* :doc:`credits/citing`
* :doc:`credits/license`
* :doc:`credits/seealso`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Credits

    credits/credits
    credits/citing
    credits/license
    credits/seealso
