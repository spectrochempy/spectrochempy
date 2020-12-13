:orphan:

.. _home:

#####################################################################
SpectroChemPy: Processing, analysing and modelling spectroscopic data
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


****************
Getting Started
****************

* :doc:`gettingstarted/whyscpy`
* :doc:`gettingstarted/overview`
* :doc:`gettingstarted/generated/auto_examples/index`
* :doc:`user/reference/install/index`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    gettingstarted/whyscpy
    gettingstarted/overview.ipynb
    Examples <gettingstarted/generated/auto_examples/index>
    Installation <user/reference/install/index>

.. _userguide:

***********************
User's Guide
***********************

* :doc:`user/userguide/introduction`
* :doc:`user/userguide/objects`
* :doc:`user/userguide/processing`
* :doc:`user/userguide/analysis`
* :doc:`user/userguide/units`
* :doc:`user/userguide/import_export`
* :doc:`user/userguide/plotting`
* :doc:`user/userguide/databases`

.. toctree::
    :maxdepth: 2
    :caption: User's Guide

    user/userguide/introduction
    user/userguide/objects
    user/userguide/processing
    user/userguide/analysis
    user/userguide/units
    user/userguide/import_export
    user/userguide/plotting
    user/userguide/databases

***********************
Reference & Help
***********************

* :doc:`user/reference/changelog`
* :doc:`user/reference/index`
* :doc:`user/reference/preference`
* :doc:`user/reference/faq`
* :doc:`user/reference/dev/issues`
* :doc:`user/reference/dev/examples`
* :doc:`user/reference/dev/contributing`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference & Help

    user/reference/changelog
    user/reference/index
    user/reference/preference
    user/reference/faq
    Bug reports & feature request <user/reference/dev/issues>
    Sharing examples <user/reference/dev/examples>
    user/reference/dev/contributing

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
