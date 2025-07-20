Import IR Data
==============

This tutorial demonstrates how to import infrared spectroscopy data in
SpectroChemPy.

.. code:: ipython3

    import spectrochempy as scp


Supported IR File Formats
-------------------------

SpectroChemPy supports these IR-specific formats:

======== ================ ==============================
Format   Function         Description
======== ================ ==============================
OMNIC    ``read_omnic()`` Thermo Scientific (.spa, .spg)
OPUS     ``read_opus()``  Bruker OPUS (.0, …)
JCAMP-DX ``read_jcamp()`` Standard IR exchange format
======== ================ ==============================


Detailled Tutorials
-------------------

.. toctree::
   :maxdepth: 3

   importOMNIC
   importOPUS
   importJCAMPDX
