:orphan:

.. _home:

#####################################################################
Processing, analyzing and modeling spectroscopic data
#####################################################################

.. toctree::
    :hidden:
    :caption: Table of Content

.. toctree::
    :hidden:
    :caption: Summary

|scpy| is a framework for processing, analyzing and modeling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. It is a cross-platform software, running on Linux, Windows or OS X.

Among its major features:

#.  Import data from experiments or modeling programs with their *metadata*
    (title, units, coordinates, ...)
#.  Preprocess these data: baseline correction, (automatic) subtraction,
    smoothing, apodization...
#.  Manipulate single or multiple datasets: concatenation, splitting, alignment
    along given dimensions, ...
#.  Explore data with exploratory analyses methods such as ``SVD``, ``PCA``, ``EFA``
    and visualization capabilities ...
#.  Modeling single or multiple datasets with curve fitting / curve modeling
    (``MCR-ALS``) methods...
#.  Export data and analyses to various formats: ``csv``, ``xls``, ``JCAMP-DX``,  ...
#.  Embed the complete workflow from raw data import to final analyses in a     Project Manager

.. only:: html

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/version.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/platforms.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/latest_release_date.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

.. warning::

    |scpy| is still experimental and under active development.
    Its current design is subject to major changes, reorganizations,
    bugs and crashes!!! Please report any issues to the
    `Issue Tracker <https://github.com/spectrochempy/spectrochempy/issues>`__

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
* :doc:`gettingstarted/papers`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    gettingstarted/whyscpy
    gettingstarted/overview
    Examples <gettingstarted/gallery/auto_examples/index>
    Installation <gettingstarted/install/index>
    gettingstarted/papers

.. _userguide:

***********************
User's Guide
***********************

* :doc:`userguide/introduction/introduction`
* :doc:`userguide/objects`
* :doc:`userguide/importexport/importexport`
* :doc:`userguide/plotting/plotting`
* :doc:`userguide/processing/processing`
* :doc:`userguide/analysis/analysis`
* :doc:`userguide/units/units`
* :doc:`userguide/databases/databases`
* :doc:`userguide/api/api`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User's Guide

    userguide/introduction/introduction
    userguide/objects
    userguide/importexport/importexport
    userguide/plotting/plotting
    userguide/processing/processing
    userguide/analysis/analysis
    userguide/units/units
    userguide/databases/databases
    userguide/api/api

***********************
Reference & Help
***********************

* :doc:`userguide/reference/changelog`
* :doc:`userguide/reference/index`
* :doc:`userguide/reference/faq`
* :doc:`devguide/issues`
* :doc:`devguide/examples`
* :doc:`devguide/index`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference & Help

    userguide/reference/changelog
    userguide/reference/index
    userguide/reference/faq
    Bug reports & feature request <devguide/issues>
    Sharing examples & tutorials <devguide/examples>
    devguide/index


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
