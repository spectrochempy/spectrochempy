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
   * - Bruker TopSpin NMR
     - Experiment directory, ``fid``, ``ser``, or processed files such as ``1r`` or ``2rr``
     - ``scp.topspin.read`` or ``scp.nmr.read_topspin``
     - :doc:`NMR </userguide/plugins/nmr>`
   * - Agilent / Varian NMR
     - Experiment directory with ``fid`` and ``procpar``
     - ``scp.agilent.read`` or ``scp.nmr.read_agilent``
     - :doc:`NMR </userguide/plugins/nmr>`
   * - JEOL NMR
     - ``.jdf``
     - ``scp.jeol.read`` or ``scp.nmr.read_jeol``
     - :doc:`NMR </userguide/plugins/nmr>`
   * - TecMag NMR
     - ``.tnt``
     - ``scp.tecmag.read`` or ``scp.nmr.read_tecmag``
     - :doc:`NMR </userguide/plugins/nmr>`
   * - SIMPSON simulation output
     - ``.spe``, ``.fid``, ``.in``
     - ``scp.nmr.read_simpson``
     - :doc:`NMR </userguide/plugins/nmr>`
   * - PerkinElmer IR
     - Single-spectrum ``.sp``
     - ``scp.perkinelmer.read``
     - :doc:`PerkinElmer </userguide/plugins/perkinelmer>`
   * - Carroucell
     - Experiment directory
     - ``scp.carroucell.read``
     - :doc:`Carroucell </userguide/plugins/carroucell>`

Use ``scp.nmr.read(...)`` to detect the format among the NMR plugin's readers.
When the format is known, the explicit reader makes the choice clear. For a
complete example from a Bruker FID to a plotted spectrum, follow the
`Bruker 1D quickstart
<https://spectrochempy-nmr.readthedocs.io/latest/getting_started/quickstart.html>`_.
