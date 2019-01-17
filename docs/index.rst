.. _main:

:version: |version| (|today|)

|scpy|: Processing, analysing and modelling data for spectrochemistry
######################################################################

|scpy| is a framework for processing, analysing and modelling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. 

It is is cross platform running on Linux, Windows or OS X.

**Main features**

#.  A |NDDataset| object embedding array of data with coordinates and metadata.
#.  A |Project| manager to work on multiple |NDDataset| simultaneoulsly.
#.  ``Units`` and ``Uncertainties`` for |NDDataset| and NDDataset coordinates |Coord|.
#.  Mathematical operations over |NDDataset| such addition, multiplication and many more ...
#.  Import functions to read data from experiments or modelling programs ...
#.  Display functions such as ``plot`` ...
#.  Export functions to ``csv``, ``xls`` ... 
#.  Preprocessing functions such as baseline correction, masking bad data, automatic subtraction 
    and many more ...
#.  Exploratory analysis such as ``svd``, ``pca``, ``mcr_als``, ``efa`` ...

.. warning::

    **This software is not yet publicly released**.
    
    |scpy| is still experimental and under active development.
    Its current design is subject to major changes, reorganizations,
    bugs and crashes!!! Please report any issues to the
    `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_

    |scpy|-0.1 is expected to be first released around june or july 2019.

Documentation
=============

**Getting Started**

* :doc:`gettingstarted/whyscpy`
* :doc:`gettingstarted/install`
* :doc:`gettingstarted/usage`
* :doc:`gettingstarted/license`
* :doc:`gettingstarted/changelog`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started
   
   gettingstarted/whyscpy
   gettingstarted/install
   gettingstarted/usage
   gettingstarted/license
   gettingstarted/changelog

   
**User Guide** 

* :doc:`user/userguide/introduction/index`
* :doc:`user/userguide/dataset/index`
* :doc:`user/userguide/projects/index`
* :doc:`user/userguide/databases/index`
* :doc:`user/userguide/nmr/index`
* :doc:`user/userguide/tutorial/index`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user/userguide/introduction/index
   user/userguide/dataset/index
   user/userguide/projects/index
   user/userguide/databases/index
   user/userguide/nmr/index
   user/userguide/tutorial/index
   
   
**Help**

* :doc:`user/api/generated/index`
* :doc:`gallery/auto_examples/index`
* :doc:`user/faq`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help

   user/api/generated/index
   gallery/auto_examples/index
   user/faq
   
   
**Developer's Corner**

* :doc:`dev/develguide`
* :doc:`dev/index`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developper's Corner

   dev/develguide
   dev/index

**Credits**

* :doc:`main/credits`
* :doc:`main/citing`
* :doc:`main/seealso`
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Credits
   
   main/credits
   main/citing
   main/seealso


