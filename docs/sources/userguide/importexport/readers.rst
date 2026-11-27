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
     - ``scp.jcamp.read`` for infrared and supported processed 1D and real 2D NMR spectra
       (requires the NMR plugin for NMR files)
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

Automatic merging
-----------------

``merge=True`` assembles observations, stacking 1D spectra or concatenating
higher-dimensional datasets along their first dimension. It does not resample
spectra. Equal array lengths alone do not establish compatibility.

Inputs must have identical ordered dimension names. Each retained dimension
must have the same coordinate representation, values and labels; all alternative
coordinates and their active selection are checked. Missing coordinates are
accepted only when absent from every input in the group. Numeric coordinates
must be finite. Units with the same physical dimensionality may be rescaled for
comparison, without changing the inputs or the retained coordinate of the first
input. Cross-dimensional spectroscopy conversions are not applied.

The comparison allows only floating-point conversion roundoff: a symmetric
relative tolerance of eight float64 machine epsilons (about ``1.8e-15``), with
zero absolute tolerance. This applies to the values exposed by ``Coord.data``,
after any configured coordinate rounding or linearization. It is not an
instrumental calibration tolerance. Shifted or reversed grids remain separate;
use an explicit alignment/resampling workflow when that is scientifically
justified.

Incompatible geometry produces a warning and separate datasets, while compatible
subsets may still be merged under the reader's origin and shape grouping rules.
Callers must handle a single ``NDDataset`` or a ``ScpObjectList``. Observation
coordinates may vary, and compatible intensity-unit conversion and mask assembly
retain their existing behavior. ``merge=False`` keeps the inputs separate.

This geometric check does not certify measurement identity. Reader-specific
restrictions remain in force; a complete policy for quantities, IR/Raman/NMR
metadata and source provenance is still outside this check. The explicit
``scp.stack()`` and ``scp.concatenate()`` operations keep their own contracts.

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
   * - NMR with format detection
     - Experiment directory or files, depending on the format
     - `scp.nmr.read <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read.html>`_
     - :doc:`NMR </userguide/plugins/nmr>`
   * - Bruker TopSpin NMR
     - Experiment directory, ``fid``, ``ser``, or processed files such as ``1r`` or ``2rr``
     - `scp.nmr.read_topspin <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_topspin.html>`_
     - :doc:`NMR </userguide/plugins/nmr>`
   * - Agilent / Varian NMR
     - Experiment directory with ``fid`` and ``procpar``
     - `scp.nmr.read_agilent <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_agilent.html>`_
     - :doc:`NMR </userguide/plugins/nmr>`
   * - JEOL NMR
     - ``.jdf``
     - `scp.nmr.read_jeol <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_jeol.html>`_
     - :doc:`NMR </userguide/plugins/nmr>`
   * - SIMPSON simulation output
     - ``.spe``, ``.fid``, ``.in``
     - `scp.nmr.read_simpson <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_simpson.html>`_
     - :doc:`NMR </userguide/plugins/nmr>`
   * - PerkinElmer IR
     - Single-spectrum ``.sp``
     - ``scp.perkinelmer.read``
     - :doc:`PerkinElmer </userguide/plugins/perkinelmer>`
   * - Carroucell
     - Experiment directory
     - ``scp.carroucell.read``
     - :doc:`Carroucell </userguide/plugins/carroucell>`

See the plugin's `guide to reading NMR data
<https://www.spectrochempy.fr/projects/nmr/latest/userguide/reading_data.html>`_
for format-specific options and examples.
