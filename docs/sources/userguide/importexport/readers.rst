.. _reader-catalog:

Reader catalog
==============

Use ``scp.read(...)`` for automatic format detection or choose an explicit
reader from the tables below. For examples of detection, protocol selection
and reading multiple files, see the :doc:`import tutorial
</userguide/importexport/import>`.

Readers included with SpectroChemPy
-----------------------------------

.. list-table::
   :header-rows: 1

   * - Format
     - Typical input
     - Reader
   * - Thermo Scientific / Nicolet OMNIC
     - ``.spa``, ``.spg``, ``.srs``
     - ``scp.omnic.read``
   * - Bruker OPUS
     - ``.0``, ``.1``, ``.000``, …
     - ``scp.opus.read``
   * - CSV
     - ``.csv``
     - ``scp.csv.read``
   * - JCAMP-DX
     - ``.dx``, ``.jdx``
     - ``scp.jcamp.read``
   * - MATLAB and Data Set Object
     - ``.mat``, ``.dso``
     - ``scp.matlab.read``
   * - HORIBA LabSpec
     - ``.txt``
     - ``scp.labspec.read``
   * - Renishaw WiRE
     - ``.wdf``
     - ``scp.wire.read``
   * - Surface Optics Corporation
     - ``.ddr``, ``.hdr``, ``.sdr``
     - ``scp.soc.read``
   * - GRAMS / Galactic
     - ``.spc``
     - ``scp.spc.read``
   * - Pfeiffer Vacuum QUADERA
     - ``.txt``
     - ``scp.quadera.read``
   * - SpectroChemPy dataset
     - ``.scp``
     - ``scp.read``
   * - ZIP archive
     - ``.zip`` containing supported files
     - ``scp.read_zip``

Use ``scp.read_dir(...)`` to scan a directory for supported files.

Readers supplied by plugins
---------------------------

Install the plugin listed for your format. SpectroChemPy discovers installed
plugins automatically. See :doc:`/userguide/plugins/official_plugins` for
installation and compatibility.

.. list-table::
   :header-rows: 1

   * - Format
     - Typical input
     - Reader
     - Required plugin
   * - NMR acquisition, processed or simulation data
     - Experiment directory or files, depending on the format
     - ``scp.nmr.read``
     - :doc:`NMR </userguide/plugins/nmr>`
   * - PerkinElmer IR
     - Single-spectrum ``.sp``
     - ``scp.perkinelmer.read``
     - :doc:`PerkinElmer </userguide/plugins/perkinelmer>`
   * - Carroucell
     - Experiment directory
     - ``scp.carroucell.read``
     - :doc:`Carroucell </userguide/plugins/carroucell>`

The NMR plugin detects the format when you call ``scp.nmr.read(...)``.
See its `guide to reading NMR data
<https://spectrochempy-nmr.readthedocs.io/latest/userguide/reading_data.html>`_
for supported formats and how to select one explicitly.
