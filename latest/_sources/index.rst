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

`SpectroChemPy` is a framework for processing, analyzing and modeling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. It is a cross-platform software, running on Linux, Windows or OS X.

Among its major features:

#.  Import data from experiments or modeling programs with their *metadata*
    (title, units, coordinates, ...)
#.  Preprocess these data: baseline correction, (automatic) subtraction,
    smoothing, apodization...
#.  Manipulate single or multiple datasets: concatenation, splitting, alignment
    along given dimensions, ...
#.  Explore data with exploratory analyses methods such as `~spectrochempy.SVD`\ ,
    `~spectrochempy.PCA`\ , `~spectrochempy.EFA` and visualization capabilities ...
#.  Modeling single or multiple datasets with curve fitting (`~spectrochempy.Optimize`\ )/ curve modeling
    (`~spectrochempy.MCR-ALS`\ ) methods...
#.  Export data and analyses to various formats: ``csv`` , ``xls`` , ``JCAMP-DX`` ,  ...
#.  Embed the complete workflow from raw data import to final analyses in a project manager

.. only:: html

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/version.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/platforms.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

   .. image:: https://anaconda.org/spectrocat/spectrochempy/badges/latest_release_date.svg
      :target: https://anaconda.org/spectrocat/spectrochempy

.. warning::

    `SpectroChemPy` is still experimental and under active development.
    Its current design is subject to major changes, reorganizations,
    bugs and crashes!!! Please report any issues to the
    `Issue Tracker <https://github.com/spectrochempy/spectrochempy/issues>`__

.. toctree::
    :hidden:

    Home <self>
    whatsnew/latest
    whatsnew/index

.. _getting_started:

****************
Getting Started
****************

* :doc:`gettingstarted/whyscpy`
* :doc:`gettingstarted/overview`
* :doc:`gettingstarted/examples/index`
* :doc:`gettingstarted/install/index`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    gettingstarted/whyscpy
    gettingstarted/overview
    Examples <gettingstarted/examples/index>
    Installation <gettingstarted/install/index>

.. _userguide:

************************
User's Guide & Tutorials
************************

The user guide is designed to give you a quick overview of the main features of SpectroChemPy. It does not cover all
features, but should help you to get started quickly, and to find your way around.
For more details on the various features, check out the :ref:`api_reference` section which gives a more detailed
description of the API. You can also refer to the :ref:`examples-index` for more examples using SpectroChemPy.

* :doc:`userguide/introduction/introduction`
* :doc:`userguide/objects/index`
* :doc:`userguide/importexport/importexport`
* :doc:`userguide/processing/processing`
* :doc:`userguide/analysis/analysis`
* :doc:`userguide/plotting/plotting`

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: User's Guide & Tutorials

    userguide/introduction/introduction
    userguide/objects/index
    userguide/importexport/importexport
    userguide/processing/processing
    userguide/analysis/analysis
    userguide/plotting/plotting

.. _reference:

***********************
Reference
***********************

* :doc:`reference/index`
* :doc:`reference/bibliography`
* :doc:`reference/glossary`
* :doc:`reference/papers`

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Reference

    reference/index
    reference/glossary
    reference/bibliography
    reference/papers

.. _contribute:

***********************
Contribute
***********************

* :doc:`devguide/issues`
* :doc:`devguide/examples`
* :doc:`devguide/index`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Contribute

    Bug reports & feature request <devguide/issues>
    Sharing examples & tutorials <devguide/examples>
    devguide/index

.. _credits:

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
