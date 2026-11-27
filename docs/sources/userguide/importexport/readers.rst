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

Recognized measurement identities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic import assembly also keeps established quantity contradictions
separate. The recognized signal-unit families are ``absorbance`` (including
``a.u.``), ``transmittance``/``absolute_transmittance``, and ``Kubelka_Munk``.
SI-prefixed forms such as ``milliabsorbance`` retain the same family. Scale
conversion within a family remains allowed, for example fractional versus
percentage transmittance. Import assembly does not implicitly convert between
these different signal families. Explicit combination and unit-conversion
operations keep their existing contracts.

Recognized scalar ``meta.jcamp_data_type`` declarations distinguish infrared,
Raman, UV-visible and NMR spectra, with ``NMR FID`` distinct from
``NMR SPECTRUM``. Contradictory known declarations keep inputs
separate, including declarations retained in native files. Existing stricter
reader restrictions, such as keeping explicitly parsed UV-visible/Raman grids
separate, still apply. A known constraint is retained through successive
assembly passes within one import, even when an intermediate assembly clears
non-common metadata.

Missing or unrecognized identities remain unknown. Vendor ``origin``, filename
extensions and free-text ``title`` values are not modality classifiers.
Generic ``percent``/``%`` and dimensionless units do not distinguish reflectance,
transmittance, or another relative signal; only the explicitly named signal
units above establish a family. User-defined units without a recognized root remain unclassified.
An unknown identity does not by itself prevent otherwise compatible assembly.

Plugins can supply additional known constraints. With the matching development
NMR plugin loaded, these include nuclear species, time/frequency domain and
decoded real/complex representation. Compatible observation-first 1D series
remain eligible; physical multidimensional NMR acquisitions stay separate,
because concatenating their first physical axis would not create a series axis.
See the plugin's `compatibility guide
<https://www.spectrochempy.fr/projects/nmr/latest/compatibility.html>`_
for its supported metadata and version requirements.

These checks do not certify complete measurement identity or preserve a full
scientific description of each source. Source provenance and identity
persistence after metadata-clearing transformations remain separate work.
Callers remain responsible for the scientific comparability of inputs.

OMNIC SPG groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPG reader checks the axis and signal definitions decoded from each member's
native header. A shared unit alone does not make wavenumber and Raman-shift axes,
or percent transmittance and percent reflectance signals, equivalent. Free-text
spectrum labels and user titles do not establish these identities; unrecognized
codes remain unknown.

By default, an inconsistent SPG group raises an error. Set
``allow_inconsistent_x=True`` to read its members separately into the import
assembly. Incompatible members remain separate even with ``merge=True``;
compatible subsets may still assemble. Add ``merge=False`` to keep every member
separate. Member timestamps and labels remain associated with their spectra,
including when sorting by date.

These SPG declarations are retained during one import, including nested ZIP or
directory reads. Persistence after save/reload and equivalent SPA/SRS checks
remain separate work.

ZIP archives and multi-dataset members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ZIP member can itself contain several datasets, for example a Galactic SPC
file with different spectral grids or a nested ZIP archive. These datasets are
returned in a flat collection alongside the other readable members. With
``merge=False``, the member datasets remain separate. With ``merge=True``, they
are offered to the same geometry and known-identity checks as other imports;
incompatible spectra stay separate without resampling or implicit signal-quantity
conversion.

The ``only`` limit counts readable members in the current archive's discovery
order. A selected member contributes all of its returned datasets, so the final
dataset count can exceed ``only``.

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
